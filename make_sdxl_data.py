import argparse
import os
import numpy as np
from PIL import Image
import random
import cv2
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from diffusers import AutoPipelineForText2Image
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import io

from ControlSketch.attn_utils import (
    cross_attn_init,
    register_cross_attention_hook,
    attn_maps,
    get_net_attn_map,
    resize_net_attn_map,
    return_net_attn_map,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:" ,device, flush=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    generator = torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return generator


def init_moedl():
    cross_attn_init()
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to("cuda")
    pipe.unet = register_cross_attention_hook(pipe.unet)
    pipe = pipe.to(device)
    return pipe

def save_compressed_npz(file_path, input_dict):
    """Save input_dict with compressed image and tensors."""
    
    # Convert PIL Image to compressed bytes
    resized_image = input_dict['image'].resize((512,512))
    img_buffer = io.BytesIO()
    resized_image.save(img_buffer, format='JPEG')
    img_data = img_buffer.getvalue()
    
    # Convert torch tensor to numpy array for efficient storage
    attn_map_np = F.interpolate(input_dict['attn_map'].unsqueeze(0).unsqueeze(0), (512, 512))[0][0].cpu().numpy()

    resized_mask = F.interpolate(input_dict['mask'].unsqueeze(0).unsqueeze(0), (512, 512))[0][0].cpu().numpy()
    binary_mask = (resized_mask > 0.5).astype(np.uint8)

    # Save using np.savez_compressed
    np.savez_compressed(
        file_path, 
        image=img_data, 
        attn_map=attn_map_np, 
        caption=input_dict['caption'], 
        mask=binary_mask,
    )


def inference_and_extract_attn(prompt, pipe, obj, generator, negative_prompt):
    image = pipe(
        prompt,
        num_inference_steps=50,
        generator=generator,
        negative_prompt=negative_prompt
    ).images[0]
    net_attn_maps = get_net_attn_map(image.size)
    net_attn_maps = resize_net_attn_map(net_attn_maps, image.size)
    net_attn_maps = return_net_attn_map(net_attn_maps, pipe.tokenizer, prompt)

    # remove sos and eos
    net_attn_maps = [attn_map for attn_map in net_attn_maps if attn_map[1].split('_')[-1] != "<<|startoftext|>>"]
    net_attn_maps = [attn_map for attn_map in net_attn_maps if attn_map[1].split('_')[-1] != "<<|endoftext|>>"]

    ind = 0
    for i, at_ in enumerate(net_attn_maps):
        if obj in at_[-1]:
            ind = i
            break
    attn = net_attn_maps[ind][0]
    attn = torch.tensor(np.array(attn))
    attn = (attn - attn.min()) / (attn.max() - attn.min())
    return image, attn


def create_masked_image(image, mask):
    image_size = image.size
    target = image.resize(image_size)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), image_size)[0][0]
    im_np = np.array(target)
    im_np = im_np / im_np.max()
    im_np = np.expand_dims(mask, axis=-1) * im_np
    im_np[mask < mask.mean()] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    masked_im = Image.fromarray(im_final).resize(image_size)
    return masked_im


def get_mask(masking_model, im: Image):
    # preprocess image
    orig_im = np.array(im)
    orig_im_size = orig_im.shape[0:2]
    im_tensor = torch.tensor(orig_im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=(1024, 1024), mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image_pre = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(device)

    with torch.no_grad():
        result = masking_model(image_pre)[0][0]
        result = result.squeeze().cpu()
    # postprocess image
    result = (result - torch.min(result)) / (torch.max(result) - torch.min(result))
    return result


def count_objects_from_tensor(image_tensor: torch.Tensor) -> int:
    # Ensure the tensor is in the range [0, 1] and has the shape (H, W)
    image_np = image_tensor.cpu().numpy() * 255  # Convert to range [0, 255]
    image_np = image_np.astype(np.uint8)

    # Convert tensor to grayscale numpy image (if not already)
    gray = image_np
    # Apply a binary threshold to get a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Number of contours found corresponds to the number of objects
    num_objects = len(contours)

    return num_objects


def get_seed_and_counter(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    file_count = len(files)
    if file_count == 0:
        return 0, 0
    highest_number = max(int(f.split('_')[-1][:-4]) for f in files)
    return highest_number, file_count


def get_obj_bb(binary_im):
    y = np.where(binary_im != 0)[0]
    x = np.where(binary_im != 0)[1]
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    return x0, x1, y0, y1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj", type=str, help="object to generate")
    # parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="",
                        help="directory to save the output dictionaries")
    parser.add_argument("--num_of_samples", type=int, default=1,
                        help="number of data samples to create for the given object")
    parser.add_argument("--save_compressed_dict", type=int, default=1, help="if 1 save compressed dictionary")
    args = parser.parse_args()
    print("Loading models", flush=True)
    pipe = init_moedl()

    masking_model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
    masking_model.to(device)

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
    model.eval()


    # subjects = [
    #     "elephant", "lion", "horse"    
    # ]

    subjects = [args.obj]

    if args.output_dir=="":
        abs_path = os.path.abspath(os.getcwd())
        args.output_dir = f"{abs_path}/SDXL_samples/"

    for obj in subjects:
        out_dir = f'{args.output_dir}/{obj}'
        print(f"Generating data for {obj}", flush=True)
        os.makedirs(out_dir, exist_ok=True)
        current_seed, counter = get_seed_and_counter(out_dir)
        counter= 0 
        while counter < args.num_of_samples:
            seed = current_seed
            generator = torch.Generator(device='cuda').manual_seed(seed)
            prompt = f"A highly detailed wide shot image of one {obj}, set against a plain mesmerizing background. center"
            negative_prompt = "close up, few, multiple"

            # image -> PIL, attn_map -> tensor range[0,1], shape [1024,1024]
            image, attn_map = inference_and_extract_attn(prompt, pipe, obj, generator, negative_prompt)
            # mask -> tensor range[0,1], shape [1024,1024]
            mask = get_mask(masking_model, image)

            num_of_objects = count_objects_from_tensor(mask)
            if num_of_objects != 1:
                print(f"Number of objects in the mask is {num_of_objects}, skipping seed {seed}", flush=True)
                current_seed += 1
                continue

            test_mask = mask.clone()
            test_mask[test_mask < 0.5] = 0
            test_mask[test_mask >= 0.5] = 1
            x0, x1, y0, y1 = get_obj_bb(test_mask)
            im_width, im_height = x1 - x0, y1 - y0
            image_area = im_width * im_height
            if (image_area < 0.2 * 1024 ** 2) or (image_area > 0.9 * 1024 ** 2):
                print(f"Object is too small, skipping seed {seed}", flush=True)
                current_seed += 1
                continue

            # create masked image
            masked_image = create_masked_image(image, mask)

            # creating captioning for the masked image
            inputs = processor(masked_image, return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            sample_dict = { "image": image,"mask": mask, "attn_map": attn_map, "caption": generated_text}
            if args.save_compressed_dict==1:
                output_path = f"{args.output_dir}/{obj}/{obj}_{seed}.npz"
                save_compressed_npz(output_path, sample_dict)
            else:
                output_path = f"{args.output_dir}/{obj}/{obj}_{seed}.npy"
                np.save(output_path, sample_dict)
            print(f"Output saved to [{output_path}]", flush=True)
            counter += 1
            current_seed += 1

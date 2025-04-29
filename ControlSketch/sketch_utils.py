import os
import io
import matplotlib.pyplot as plt
import numpy as np
import pydiffvg
import skimage
import skimage.io
import torch
import wandb
from PIL import Image
from torchvision.utils import make_grid
import subprocess as sp
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from skimage.transform import resize
from scipy.ndimage import  binary_erosion, binary_dilation




def imwrite(img, filename, gamma=2.2, normalize=False, use_wandb=False, wandb_name="", step=0, input_im=None):
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        # repeat along the third dimension
        img = np.expand_dims(img, 2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0 / gamma)
    img = (img * 255).astype(np.uint8)

    skimage.io.imsave(filename, img, check_contrast=False)
    images = [wandb.Image(Image.fromarray(img), caption="output")]
    if input_im is not None and step == 0:
        images.append(wandb.Image(input_im, caption="input"))
    if use_wandb:
        wandb.log({wandb_name + "_": images}, step=step)


def save_mask(mask, save_path, use_wandb):
    mask_save = mask.clone()
    mask_save[mask_save < 0.5] = 0
    mask_save[mask_save >= 0.5] = 1

    h, w = mask_save.shape[-2:]  # Ensure H and W
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    
    plt.imshow(mask_save, cmap='gray')
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(save_path, 'mask.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
    if use_wandb:
        wandb.log({"mask": wandb.Image(plt)})
    plt.close(fig)


def load_svg(path_svg):
    svg = os.path.join(path_svg)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        svg)
    return canvas_width, canvas_height, shapes, shape_groups


def read_svg(path_svg, device, multiply=False, args=None):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        path_svg)
    if args is not None and (hasattr(args, 'scale_w') or hasattr(args, 'scale_h')):
        w, h = args.scale_w, args.scale_h
        for path in shapes:
            path.points = path.points / canvas_width
            path.points = 2 * path.points - 1
            path.points[:, 0] /= (w)  # / canvas_width)
            path.points[:, 1] /= (h)  # / canvas_height)
            path.points = 0.5 * (path.points + 1.0) * canvas_width
            center_x, center_y = canvas_width / 2, canvas_height / 2
            path.points[:, 0] += (args.original_center_x * canvas_width - center_x)
            path.points[:, 1] += (args.original_center_y * canvas_height - center_y)

    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,  # num_samples_x
                  2,  # num_samples_y
                  0,  # seed
                  None,
                  *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
          torch.ones(img.shape[0], img.shape[1], 3,
                     device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    return img

def get_obj_bb(binary_im):
    y = np.where(binary_im != 0)[0]
    x = np.where(binary_im != 0)[1]
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    return x0, x1, y0, y1


def cut_and_resize(im, x0, x1, y0, y1, new_height, new_width, type):
    cut_obj = im[y0: y1, x0: x1]
    resized_obj = resize(cut_obj, (new_height, new_width))
    if type== "mask":
        new_mask = np.zeros(im.shape)
    else: #type == image
        new_mask = np.ones(im.shape)
    center_y_new = int(new_height / 2)
    center_x_new = int(new_width / 2)
    center_targ_y = int(new_mask.shape[0] / 2)
    center_targ_x = int(new_mask.shape[1] / 2)
    startx, starty = center_targ_x - center_x_new, center_targ_y - center_y_new
    new_mask[starty: starty + resized_obj.shape[0], startx: startx + resized_obj.shape[1]] = resized_obj
    return new_mask


def increase_object_size(renderer, scale_w, scale_h, original_center_x, original_center_y):
       #Increases the size of the object on the canvas to its original size
       with torch.no_grad():
            w, h = scale_w, scale_h
            canvas_width, canvas_height = 512, 512
            for path in renderer.shapes:
                path.points = path.points / canvas_width
                path.points = 2 * path.points - 1
                path.points[:, 0] /= (w)  
                path.points[:, 1] /= (h)  
                path.points = 0.5 * (path.points + 1.0) * canvas_width
                center_x, center_y = canvas_width / 2, canvas_height / 2
                path.points[:, 0] += (original_center_x * canvas_width - center_x)
                path.points[:, 1] += (original_center_y * canvas_height - center_y)


def resize_svg(renderer, target_width, target_height):
    original_width= original_height = 512
    # Calculate scaling factors
    width_scale = target_width / original_width
    height_scale = target_height / original_height

    # Update renderer's canvas size
    renderer.canvas_width = target_width
    renderer.canvas_height = target_height

    with torch.no_grad():
        # Update the control points of each path
        for shape in renderer.shapes:
            for i, point in enumerate(shape.points):
                x, y = point[0], point[1]
                # Scale the points
                shape.points[i] = torch.tensor([x * width_scale, y * height_scale], device=point.device)
            # Scale the stroke width
            new_width = float(shape.stroke_width) * float(width_scale)
            shape.stroke_width = torch.tensor(new_width)

    return renderer


def make_video(cur_path):
    output_width = 224
    output_height = 224

    sp.run(["ffmpeg", "-y", "-framerate", "10", "-pattern_type", "glob", "-i",
            f"{cur_path}/svg_to_png/iter_*.png", "-vb", "20M",
            "-vf", f"scale={output_width}:{output_height}",  # Specify output size
            f"{cur_path}/sketch.mp4"])
    


def plot_initial_points(attn, clustered_mask, inputs, inds, use_wandb, output_path):
  
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 0], inds[:, 1], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("attn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(clustered_mask.astype(np.uint8))  # Use a categorical color map
    plt.title("clustered mask")
    plt.scatter(inds[:, 0], inds[:, 1], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"initial_points": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()




def create_masked_image(image, mask):
    # Convert the image to a numpy array and normalize
    im_np = np.array(image)
    im_np = im_np / im_np.max()

     # Apply mask to the image
    im_np = np.expand_dims(mask, axis=-1) * im_np
    im_np[mask < mask.mean()] = 1

    # Convert back to an image
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    masked_im = Image.fromarray(im_final)
    return masked_im 


 
def fix_image_scale(im):
    im_np = np.array(im) / 255
    height, width = im_np.shape[0], im_np.shape[1]
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    new_background[y: y + height, x: x + width] = im_np
    new_background = (new_background / new_background.max()
                      * 255).astype(np.uint8)
    new_im = Image.fromarray(new_background)
    return new_im



def fix_image_mask_scale(im, mask):
    # Convert image to numpy array and scale pixel values
    im_np = np.array(im) / 255
    mask_np = np.array(mask)

    # Get image and mask dimensions
    height, width = im_np.shape[0], im_np.shape[1]
    
    # Determine the dimensions for the new background (same for image and mask)
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    new_mask = np.zeros((max_len, max_len))  # Create a new mask background filled with zeros
    
    # Calculate the centering offset for the image and mask
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    
    # Place the image and mask onto their respective backgrounds
    new_background[y: y + height, x: x + width] = im_np
    new_mask[y: y + height, x: x + width] = mask_np

    # Normalize and convert the new background to 0-255 scale
    new_background = (new_background * 255).astype(np.uint8)
    
    # Convert numpy arrays back to Image objects
    new_im = Image.fromarray(new_background)
    new_im_mask = torch.from_numpy(new_mask)
    
    return new_im, new_im_mask





def get_mask(im: Image, device):
    '''
    Uses bria model to extract the object mask
    '''
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)
    model.to(device)

    # preprocess image
    orig_im = np.array(im)
    orig_im_size = orig_im.shape[0:2]
    im_tensor = torch.tensor(orig_im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=orig_im_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image_pre = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0]).to(device)
    
    with torch.no_grad():
        result=model(image_pre)[0][0]
        result = result.squeeze().cpu()
    # postprocess image
    result = (result - torch.min(result)) / (torch.max(result) - torch.min(result))
    return result



def load_compressed_npz(file_path):
    """Load compressed .npz file and reconstruct the original objects."""
    data = np.load(file_path, allow_pickle=True)

    # Reconstruct the image from bytes
    img_bytes = data["image"].tobytes()
    image = Image.open(io.BytesIO(img_bytes))

    result = {"image": image}

    if "mask" in data:
        result["mask"] = torch.from_numpy(data["mask"]).float()
    if "attn_map" in data:
        result["attn_map"] = torch.from_numpy(data["attn_map"])
    if "caption" in data:
        result["caption"] = data["caption"].item()

    return result


def get_thick_contour_tensor(mask, canvas_width, canvas_height):
        # Resize to (canvas_width, canvas_height)
        mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(canvas_width, canvas_height), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        # mask to binary
        mask_resized[mask_resized < 0.5] = 0
        mask_resized[mask_resized >= 0.5] = 1

            # Convert tensor to NumPy array
        binary_array = mask_resized.numpy()
        
        # Perform binary erosion (This shrinks the foreground region, leaving the core area.)
        eroded = binary_erosion(binary_array, structure=np.ones((10, 10)))
        
        # Perform binary dilation (This expands the foreground region, enlarging the boundary.)
        dilated = binary_dilation(binary_array, structure=np.ones((5, 5)))
        
        # Find the thick contour
        thick_contour = dilated.astype(np.uint8) - eroded.astype(np.uint8)
        
        # Convert back to PyTorch tensor
        thick_contour_tensor = torch.tensor(thick_contour)
        return thick_contour_tensor
    
        
def sort_by_contour_and_attn(renderer, mask, attn_map):

    with torch.no_grad():

        # Use pydiffvg to parse the SVG directly from file
        canvas_width, canvas_height, shapes, shape_groups = renderer.canvas_width, renderer.canvas_height, renderer.shapes,renderer.shape_groups

        # Render each stroke separately to count intersecting pixels
        stroke_to_pixels = {}  # Map stroke index to its pixel locations
        intersection_pixel_count = []  # To store number of intersecting pixels

        thick_contour_tensor = get_thick_contour_tensor(mask, canvas_width, canvas_height)
        contour_mask = thick_contour_tensor.numpy()

        attn_resized = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), size=(canvas_width, canvas_height), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        attn_map = attn_resized.numpy()

        for i, shape in enumerate(shapes):
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                            fill_color=None,
                                            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
            single_shape_group = [path_group]

            scene_args = pydiffvg.RenderFunction.serialize_scene(
                canvas_width, canvas_height, [shape], single_shape_group
            )
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

            img = img[:, :, 3:4] * img[:, :, :3] + \
                torch.ones(img.shape[0], img.shape[1], 3, device=pydiffvg.device) * (1 - img[:, :, 3:4])
            img = img[:, :, :3].cpu().numpy()
            mask = ~np.any(img > 0, axis=-1)  # Binary mask for stroke pixels

            stroke_to_pixels[i] = mask

            intersection_pixels = np.logical_and(mask, contour_mask)
            intersection_count = np.sum(intersection_pixels)
            intersection_pixel_count.append((i, intersection_count))

        sorted_non_zero_intersection_strokes = sorted([(i, count) for i, count in intersection_pixel_count if count > 0], key=lambda x: x[1], reverse=True)
        zero_intersection_strokes = [(i, count) for i, count in intersection_pixel_count if count == 0]

        avg_attention_values = []
        for stroke_index, _ in zero_intersection_strokes:
            shape = shapes[stroke_index]
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([0]),
                fill_color=None,
                stroke_color=torch.tensor([1.0, 1.0, 1.0, 1.0])
            )
            single_shape_group = [path_group]

            scene_args = pydiffvg.RenderFunction.serialize_scene(
                canvas_width, canvas_height, [shape], single_shape_group
            )
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

            img = img[:, :, 3:4] * img[:, :, :3] + \
                torch.zeros(img.shape[0], img.shape[1], 3, device=pydiffvg.device) * (1 - img[:, :, 3:4])
            img = img[:, :, :3].cpu().numpy()
            stroke_mask = np.any(img > 0.9, axis=-1)

            num_stroke_pixels = np.sum(stroke_mask)

            stroke_attention_values = attn_map * stroke_mask
            total_attention_value = np.sum(stroke_attention_values)
            avg_attention_value = total_attention_value / num_stroke_pixels if num_stroke_pixels > 0 else 0

            avg_attention_values.append((stroke_index, avg_attention_value))

        sorted_zero_intersection_strokes = sorted(avg_attention_values, key=lambda x: x[1], reverse=True)

        final_sorted_strokes = sorted_non_zero_intersection_strokes + sorted_zero_intersection_strokes

        sorted_indices = [i for i, _ in final_sorted_strokes]
        sorted_shapes = []
        sorted_shape_groups = []
        for j in sorted_indices:
            sorted_shapes.append(shapes[j])
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(sorted_shapes) - 1]),
                                            fill_color=None,
                                            stroke_color=shape_groups[j].stroke_color)
            sorted_shape_groups.append(path_group)
        

        renderer.shapes = sorted_shapes
        renderer.shape_groups= sorted_shape_groups


          



        













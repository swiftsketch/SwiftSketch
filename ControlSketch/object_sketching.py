import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import os
import sys
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from control_sds_loss_file import ControlSDSLoss
import config
import sketch_utils as utils
from painter_params import Painter, PainterOptimizer


def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_strokes, args=args,
                       num_segments=args.num_segments,
                       device=args.device,
                       target_im=target_im,
                       mask=mask,
                       )
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    if args.target_is_dict:  # in case a dictionary is provided
        if os.path.splitext(os.path.basename(args.target))[-1] == ".npy":
            input_dict = np.load(args.target, allow_pickle='TRUE').item()
        else:
            input_dict= utils.load_compressed_npz(args.target)
        target = input_dict["image"]
        target = target.resize((args.render_size, args.render_size))
        if "mask" in input_dict:
            mask = input_dict["mask"] 
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]
        else: 
            mask = utils.get_mask(target, args.device)
        if not args.caption and "caption" in input_dict:
            args.caption = input_dict["caption"]
        target= utils.create_masked_image(target, mask)
        if "attn_map" in input_dict:
            attn = input_dict["attn_map"]
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]
            args.attn_from_dict= attn

    else:
        target = Image.open(args.target)
        if target.mode == "RGBA":
            # Create a white rgba background
            new_image = Image.new("RGBA", target.size, "WHITE")
            # Paste the image on the background.
            new_image.paste(target, (0, 0), target)
            target = new_image
        target = target.convert("RGB")
        mask = utils.get_mask(target, args.device)
        target = utils.create_masked_image(target, mask)
        if args.fix_scale:
            target, mask = utils.fix_image_mask_scale(target, mask)
        target = target.resize((args.render_size, args.render_size))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]
    
    # Reduces the size of the object on the canvas if needed
    im_np = np.array(target)
    test_mask = mask.clone()
    test_mask[test_mask < 0.5] = 0
    test_mask[test_mask >= 0.5] = 1
    w, h = target.size[0], target.size[1]
    x0, x1, y0, y1 = utils.get_obj_bb(test_mask)
    im_width, im_height = x1 - x0, y1 - y0
    max_size = max(im_width, im_height)
    target_size = int(args.render_size * args.object_size_ratio)
    if max_size > target_size: 
        if im_width > im_height:
            new_width, new_height = target_size, int((target_size / im_width) * im_height)
        else:
            new_width, new_height = int((target_size / im_height) * im_width), target_size
        mask_np3 = np.stack([test_mask] * 3, axis=-1)
        mask = utils.cut_and_resize(mask_np3, x0, x1, y0, y1, new_height, new_width, "mask")
        mask = torch.from_numpy(mask[:, :, 0])  
        target_np = im_np / im_np.max()
        im_np = utils.cut_and_resize(target_np, x0, x1, y0, y1, new_height, new_width, "image")
        im_np_final = (im_np / im_np.max() * 255).astype(np.uint8)
        target= Image.fromarray(im_np_final)
        

        args.new_width= new_width
        args.new_height= new_height
        args.obj_bb= (x0, x1, y0, y1)
        args.original_center_y = (y0 + (y1 - y0) / 2) / h
        args.original_center_x = (x0 + (x1 - x0) / 2) / w
        args.scale_w = new_width / im_width
        args.scale_h = new_height / im_height

    
    args.input_image = target
    args.mask = mask
    target.save(f"{args.output_dir}/input.png")
    if args.use_wandb:
        wandb.log({"input": wandb.Image(target)})
    utils.save_mask(mask, args.output_dir, args.use_wandb)
    data_transform = transforms.ToTensor()
    target = data_transform(target).unsqueeze(0).to(args.device)
    return target, mask



def main(args):
    print("run object sketching", flush=True)
    inputs, mask = get_target(args)  
    renderer = load_renderer(args, inputs, mask) 
    sds_loss = ControlSDSLoss(args, args.device)

    optimizer = PainterOptimizer(args, renderer)
    img = renderer.init_image()  
    optimizer.init_optimizers()  
    print("Starting the optimization process", flush=True)
    
    epoch_range = tqdm(range(args.num_iter+1))

    inputs = inputs.detach()
    for epoch in epoch_range:
        optimizer.zero_grad_()
        sketches = renderer.get_image().to(args.device)  
        loss = sds_loss(sketches)
        loss.backward()
        optimizer.step_()


        if epoch % args.save_interval == 0:  # save current sketch
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
            if not os.path.exists(f"{args.output_dir}/svg_to_png"):
                os.mkdir(f"{args.output_dir}/svg_to_png")
            path_svg = f"{args.output_dir}/svg_logs/svg_iter{epoch}.svg"
            sketch_iter = utils.read_svg(path_svg, args.device, multiply=True, args=args).cpu().numpy()
            sketch_iter = Image.fromarray((sketch_iter * 255).astype('uint8'), 'RGB')
            sketch_iter.save("{0}/{1}/iter_{2:04}.png".format(args.output_dir, "svg_to_png", int(epoch)))
            if args.use_wandb:
                sketch_array = np.array(sketch_iter)
                wandb.log({"cur_sketch": wandb.Image(sketch_array)}, step=epoch)


        if epoch == 0 and args.use_init_method:
            utils.plot_initial_points(renderer.get_attn_map_to_plot(), renderer.get_clustered_mask(), inputs, renderer.get_inds(),
                            args.use_wandb, "{}/{}.jpg".format(
                args.output_dir, "initial_points"))


    # save final sketch
    if args.sort_final_sketch:
        utils.sort_by_contour_and_attn(renderer, args.mask, renderer.get_attn())
    if (hasattr(args, 'scale_w') or hasattr(args, 'scale_h')): 
        #Increases the size of the object on the canvas to its original size if it has been reduced
        utils.increase_object_size(renderer, args.scale_w, args.scale_h, args.original_center_x, args.original_center_y)
    if args.output_svg_size!=512:
        utils.resize_svg(renderer, args.output_svg_size, args.output_svg_size)
    renderer.save_svg(args.output_dir, "final_svg")
    final_sketch_num = utils.read_svg(f"{args.output_dir}/final_svg.svg", args.device, multiply=True,
                                      args=None).cpu().numpy()
    final_sketch = Image.fromarray((final_sketch_num * 255).astype('uint8'), 'RGB')
    final_sketch.save(f"{args.output_dir}/final_sketch.png")
    print(f"You can download the result sketch from {args.output_dir}/final_sketch.png")
    if args.use_wandb:
        final_sketch = np.array(final_sketch)
        wandb.log({f"final sketch": wandb.Image(final_sketch)})


    # save video
    utils.make_video(args.output_dir)
    if args.use_wandb:
        video_path = f"{args.output_dir}/sketch.mp4"
        wandb.log({"sketch_video": wandb.Video(video_path, format="mp4")})


    if args.target_is_dict  and args.save_svg_in_dict:
        # saves the result svg into the dict
        final_key = f'svg_{args.num_strokes}s'
    
        with open(f"{args.output_dir}/final_svg.svg", 'r') as svg_file:
            svg_content = svg_file.read()

        if os.path.splitext(os.path.basename(args.target))[-1] == ".npy":
            data = np.load(args.target, allow_pickle=True).item()
            data[final_key] = svg_content
            np.save(args.target, data)
        else:
            data = dict(np.load(args.target, allow_pickle=True))
            data[final_key] = svg_content
            np.savez_compressed(args.target, **data)



            
        print("The final SVG was saved to the input dictionary")


if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()


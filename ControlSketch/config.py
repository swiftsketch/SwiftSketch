import argparse
import os
import random
import numpy as np
import pydiffvg
import torch
import wandb



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser()

    # =================================
    # ============ general ============
    # =================================
    parser.add_argument("--target", help="target image or npy/npz dict path")
    parser.add_argument("--save_svg_in_dict", type=int, default=1, help="if 1 and target_is_dict is 1, save the final svg in the dict") 
    parser.add_argument("--output_dir", type=str, default="",
                        help="directory to save the output images")
    parser.add_argument("--use_cpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fix_scale", type=int, default=0, help="if the target image is not squared, it is recommended to fix the scale")
    parser.add_argument("--sort_final_sketch", type=int, default=1, help="sort the strokes in the final SVG file")

    # =================================
    # ============ wandb ============
    # =================================
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_user", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="defualt")
    parser.add_argument("--wandb_project_name", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
   
    # =================================
    # =========== training ============
    # =================================
    parser.add_argument("--num_iter", type=int, default=2000,
                        help="number of optimization iterations")
    parser.add_argument("--lr_scheduler", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="for optimization it's only one image")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--object_size_ratio", type=float, default=0.75)     
    parser.add_argument("--render_size", type=int, default=512)   
    parser.add_argument("--output_svg_size", type=int, default=512) 
    

    # =================================
    # ======== strokes params =========
    # =================================
    parser.add_argument("--num_strokes", type=int,
                        default=32, help="number of strokes used to generate the sketch, this defines the level of abstraction")
    parser.add_argument("--width", type=float,
                        default=2.5, help="stroke width")
    parser.add_argument("--control_points_per_seg", type=int, default=4)
    parser.add_argument("--num_segments", type=int, default=1,
                        help="number of segments for each stroke, each stroke is a bezier curve with 4 control points")
   
    # ====================================================
    #== attention map for strokes initialization params ==
    # ====================================================
    parser.add_argument("--use_init_method", type=int, default=1,
                        help="if True, use the initialization method to set the location of the initial strokes, and not random")
    parser.add_argument("--object_name", type=str, default="", help="the word for extrcting the object attention map")
    parser.add_argument("--attn_model", type=str,  default="diffusion", choices=["diffusion", "clip"], help="Choose between 'diffusion' and 'clip'")

    # =================================
    # ============= control_net sds loss ==============
    # =================================
    parser.add_argument("--diffusion_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--diffusion_guidance_scale", type=int, default=100)
    parser.add_argument("--caption", type=str, default="")
    parser.add_argument("--conditioning_scale", type=float, default=0.15)
    parser.add_argument("--condition", type=str, default="depth", choices=["depth", "hed", "scribble", "seg", "canny", "normal"], 
                        help="Choose between depth, hed, scribble, seg, canny, normal")
   

    args = parser.parse_args()
    set_seed(args.seed)

    assert os.path.isfile(args.target), f"{args.target} does not exists!"

    args.target_is_dict= os.path.splitext(os.path.basename(args.target))[-1] == ".npy" or os.path.splitext(os.path.basename(args.target))[-1] == ".npz"

    if args.output_dir=="":
        abs_path = os.path.abspath(os.getcwd())
        args.output_dir = f"{abs_path}/output_sketches/"

    test_name = os.path.splitext(os.path.basename(args.target))[0]
    output_dir = f"{args.output_dir}/{test_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    args.output_dir_target=output_dir

    if args.wandb_name=="defualt":
        args.wandb_name = f"{test_name}_{args.num_strokes}_strokes"

    args.output_dir = os.path.join(output_dir, args.wandb_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    jpg_logs_dir = f"{args.output_dir}/jpg_logs"
    svg_logs_dir = f"{args.output_dir}/svg_logs"
    if not os.path.exists(jpg_logs_dir):
        os.mkdir(jpg_logs_dir)
    if not os.path.exists(svg_logs_dir):
        os.mkdir(svg_logs_dir)

    if args.use_wandb:
       wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                   config=args, name=args.wandb_name, id=wandb.util.generate_id())
       
    use_gpu = not args.use_cpu
    if not torch.cuda.is_available():
        use_gpu = False
        print("CUDA is not configured with GPU, running with CPU instead.")
    if use_gpu:
        args.device = torch.device("cuda" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    else:
        args.device = torch.device("cpu")

    pydiffvg.set_use_gpu(torch.cuda.is_available() and use_gpu)
    pydiffvg.set_device(args.device)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    final_config = vars(args)
    np.save(f"{args.output_dir}/config_init.npy", final_config)
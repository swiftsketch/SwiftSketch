
import torch
import numpy as np
from PIL import Image
from controlnet_aux import HEDdetector, NormalBaeDetector, MidasDetector 
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
import cv2
from diffusers import ControlNetModel


def controlnet(condition, device):
    if condition== "depth":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16, use_safetensors=True).to(device)
    elif condition== "hed":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16, use_safetensors=True).to(device)
    elif condition== "scribble":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16, use_safetensors=True).to(device)
    elif condition== "seg":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16, use_safetensors=True).to(device)
    elif condition== "canny": 
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True).to(device)
    elif condition== "normal":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", torch_dtype=torch.float16, use_safetensors=True).to(device)
    return controlnet


def controlnet_11(condition, device):
    if condition== "depth":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, use_safetensors=True).to(device)
    elif condition== "hed":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_softedge", torch_dtype=torch.float16, use_safetensors=True).to(device)
    elif condition== "scribble":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_scribble", torch_dtype=torch.float16, use_safetensors=True).to(device)
    elif condition== "seg":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16, use_safetensors=True).to(device)
    elif condition== "canny": 
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16, use_safetensors=True).to(device)
    elif condition== "normal":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", torch_dtype=torch.float16, use_safetensors=True).to(device)
    return controlnet


def create_condition(image, condition):
    if condition== "depth":
        condition= create_depth_condition(image)
    elif condition== "hed":
        condition= create_hed_condition(image)
    elif condition== "scribble":
        condition= create_hed_condition(image)
    elif condition== "seg":
        condition= create_seg_condition(image)
    elif condition== "canny":
        condition= create_canny_condition(image)
    elif condition== "normal":
        condition= create_normal_condition(image)
    return condition


def create_normal_condition(image):
    normal_processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
    control_image = normal_processor(image)
    return control_image

def create_canny_condition(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

  
def create_depth_condition(image):
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
    image= midas(image) 
    return image


def create_hed_condition(image): 
    hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
    image = hed(image)
    return image


palette = np.asarray([
[0, 0, 0],[120, 120, 120],[180, 120, 120],[6, 230, 230],[80, 50, 50],[4, 200, 3],[120, 120, 80],[140, 140, 140],
[204, 5, 255],[230, 230, 230],[4, 250, 7],[224, 5, 255],[235, 255, 7],[150, 5, 61],[120, 120, 70],[8, 255, 51],
[255, 6, 82],[143, 255, 140],[204, 255, 4],[255, 51, 7],[204, 70, 3],[0, 102, 200],[61, 230, 250],[255, 6, 51],[11, 102, 255],
[255, 7, 71],[255, 9, 224],[9, 7, 230],[220, 220, 220],[255, 9, 92],[112, 9, 255],[8, 255, 214],[7, 255, 224],[255, 184, 6],
[10, 255, 71],[255, 41, 10],[7, 255, 255],[224, 255, 8],[102, 8, 255],[255, 61, 6],[255, 194, 7],[255, 122, 8],[0, 255, 20],
[255, 8, 41],[255, 5, 153],[6, 51, 255],[235, 12, 255],[160, 150, 20],[0, 163, 255],[140, 140, 140],[250, 10, 15],
[20, 255, 0],[31, 255, 0],[255, 31, 0],[255, 224, 0],[153, 255, 0],[0, 0, 255],[255, 71, 0],[0, 235, 255],[0, 173, 255],
[31, 0, 255],[11, 200, 200],[255, 82, 0],[0, 255, 245],[0, 61, 255],[0, 255, 112],[0, 255, 133],[255, 0, 0],[255, 163, 0],
[255, 102, 0],[194, 255, 0],[0, 143, 255],[51, 255, 0],[0, 82, 255],[0, 255, 41],[0, 255, 173],[10, 0, 255],[173, 255, 0],
[0, 255, 153],[255, 92, 0],[255, 0, 255],[255, 0, 245],[255, 0, 102],[255, 173, 0],[255, 0, 20],[255, 184, 184],[0, 31, 255],
[0, 255, 61],[0, 71, 255],[255, 0, 204],[0, 255, 194],[0, 255, 82],[0, 10, 255],[0, 112, 255],[51, 0, 255],[0, 194, 255],
[0, 122, 255],[0, 255, 163],[255, 153, 0],[0, 255, 10],[255, 112, 0],[143, 255, 0],[82, 0, 255],[163, 255, 0],[255, 235, 0],
[8, 184, 170],[133, 0, 255],[0, 255, 92],[184, 0, 255],[255, 0, 31],[0, 184, 255],[0, 214, 255],[255, 0, 112],[92, 255, 0],
[0, 224, 255],[112, 224, 255],[70, 184, 160],[163, 0, 255],[153, 0, 255],[71, 255, 0],[255, 0, 163],[255, 204, 0],[255, 0, 143],
[0, 255, 235],[133, 255, 0],[255, 0, 235],[245, 0, 255],[255, 0, 122],[255, 245, 0],[10, 190, 212],[214, 255, 0],
[0, 204, 255],[20, 0, 255],[255, 255, 0],[0, 153, 255],[0, 41, 255],[0, 255, 204],[41, 0, 255],[41, 255, 0],
[173, 0, 255],[0, 245, 255],[71, 0, 255],[122, 0, 255],[0, 255, 184],[0, 92, 255],[184, 255, 0],[0, 133, 255],[255, 214, 0],
[25, 194, 194],[102, 255, 0],[92, 0, 255],
])


    

def create_seg_condition(image):
    image_processor_for_seg = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor_for_seg = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
    pixel_values = image_processor_for_seg(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = image_segmentor_for_seg(pixel_values)

    seg = image_processor_for_seg.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)
    print("color_seg.shape")
    print(color_seg.shape)

    image = Image.fromarray(color_seg)
    return image
    
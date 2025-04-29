import random
import CLIP_.clip as clip
import numpy as np
import pydiffvg
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.cluster import KMeans
import sketch_utils as utils
import inversion
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
import matplotlib.pyplot as plt
from attn_utils import (
    upscale,
    resize_net_attn_map,
    return_net_attn_map,
)



class Painter(torch.nn.Module):
    def __init__(self, args,
                 num_strokes=4,
                 num_segments=4,
                 device=None,
                 target_im=None,
                 mask=None,
                 ):
        super(Painter, self).__init__()

        self.args = args
        self.render_size = args.render_size
        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg
        self.mask = mask
        self.strokes_counter = 0  
        self.shapes = []
        self.shape_groups = []
        self.device = device
        self.canvas_width, self.canvas_height = args.render_size, args.render_size
        self.points_vars = []
        self.optimize_flag = []
        self.initial_points = []

        # attention related for strokes initialisation
        self.use_init_method = args.use_init_method
        self.target_im= target_im
        self.target_path = args.target
        self.attn_model = args.attn_model
        self.attention_map = self.set_attention_map() if self.use_init_method else None
        self.attn_map_to_plot = self.set_attention_threshold_map() if self.use_init_method else None
        

    def init_image(self):
        for i in range(self.num_paths):  
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path = self.get_path() 
            self.shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                                fill_color=None,
                                                stroke_color=stroke_color)
            self.shape_groups.append(path_group)
        self.optimize_flag = [True for i in range(len(self.shapes))]

        img = self.render_warp() 
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (
                1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # HWC -> NCHW
        return img
        

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # HWC -> NCHW
        return img

    def get_path(self):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        p0 = self.inds_normalised[self.strokes_counter] if self.use_init_method else (random.random(), random.random())
        self.initial_points.append(p0)
        points.append(p0) 
        for j in range(self.num_segments):  # here is 1 by defult
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                self.initial_points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height

        path = pydiffvg.Path(num_control_points=self.num_control_points,
                             points=points,
                             stroke_width=torch.tensor(self.width),
                             is_closed=False)  

        self.strokes_counter += 1
        return path

    def render_warp(self):  
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = _render(self.canvas_width,  
                      self.canvas_height, 
                      2,  
                      2,  
                      0,  
                      None,
                      *scene_args)
        return img

    def parameters(self):
        self.points_vars = []
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.points_vars.append(path.points)
        return self.points_vars

    def get_points_parans(self):
        return self.points_vars
    

    def save_svg(self, output_dir, name):
        pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes,
                          self.shape_groups)

    def get_initial_points(self):
        return torch.tensor(self.initial_points)

 

    def define_clip_attention_input(self, target_im):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
            preprocess.transforms[-1],
        ])
        image_input_attn_clip= data_transforms(target_im).to(self.device)
        image_input_attn_clip = F.interpolate(image_input_attn_clip, size=(224, 224), mode='bicubic', align_corners=False)
        self.image_input_attn_clip = image_input_attn_clip
    
    def interpret(self, image, model, device):
        images = image.repeat(1, 1, 1, 1)
        res = model.encode_image(images)
        model.zero_grad()
        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
        cams = []  # there are 12 attention blocks
        for i, blk in enumerate(image_attn_blocks):
            cam = blk.attn_probs.detach()  # attn_probs shape is 12, 50, 50
            # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
            cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0)
            cam = cam.clamp(min=0).mean(dim=1)  # mean of the 12 something
            cams.append(cam)
            R = R + torch.bmm(cam, R)

        cams_avg = torch.cat(cams)  # 12, 50, 50
        cams_avg = cams_avg[:, 0, 1:]  # 12, 1, 49
        image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
        image_relevance = image_relevance.reshape(1, 1, 7, 7)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
        image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        return image_relevance



    def clip_attn(self):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        attn_map = self.interpret(self.image_input_attn_clip, model, device=self.device)
        attn_map = torch.from_numpy(attn_map)
        del model
        torch.cuda.empty_cache()
        return attn_map
    
 
    

    def diffusion_attn(self):
        # DDIM inversion
        num_inference_steps = 50
        # orig_image= Image.open(self.args.target).resize((1024, 1024))
        orig_image= self.args.input_image.resize((1024, 1024))
        
        x0 = np.array(orig_image)
        caption = f"a portrait of a {self.args.object_name}"

        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False)

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True,
            scheduler=scheduler
        ).to(self.device)


        zts = inversion.ddim_inversion(pipeline, x0, caption, num_inference_steps, 2)

        zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)

        # Register the custom attention processor and get the attn_maps list
        pipeline.unet, attn_maps = register_attention_store(pipeline.unet)
        pipeline = pipeline.to(self.device)

        g_cpu = torch.Generator(device='cpu')
        g_cpu.manual_seed(10)

        latents = torch.randn(1, 4, 128, 128, device='cpu', generator=g_cpu,
                            dtype=pipeline.unet.dtype, ).to(self.device)
        
        latents[0] = zT

        image = pipeline(caption, latents=latents,
                        callback_on_step_end=inversion_callback,
                        num_inference_steps=num_inference_steps, guidance_scale=10.0).images[0]

        attn_map = inference_and_extract_attn(attn_maps, caption, pipeline, image, self.args.object_name)
        attn_map= torch.pow(attn_map, 2)
        
        del latents, zts, zT, image, attn_maps, inversion_callback
        del pipeline
        torch.cuda.empty_cache()
        return attn_map


    def set_attention_map(self):
        if hasattr(self.args, 'attn_from_dict'): 
            attn = self.args.attn_from_dict
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (self.render_size, self.render_size))[0][0]
            if hasattr(self.args, 'obj_bb'): 
                attn = np.stack([attn] * 3, axis=-1)
                x0, x1, y0, y1= self.args.obj_bb
                attn = utils.cut_and_resize(attn, x0, x1, y0, y1, self.args.new_height, self.args.new_width, "mask")
                attn = torch.from_numpy(attn[:, :, 0])
        elif self.attn_model == "clip" or self.args.object_name == "" :
            self.saliency_clip_model = "ViT-B/32"
            self.define_clip_attention_input(self.target_im)
            attn= self.clip_attn()
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (self.render_size, self.render_size))[0][0]
        else: # self.attn_model == "diffusion":
            attn= self.diffusion_attn()
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (self.render_size, self.render_size))[0][0]
        return attn
    
    def weighted_kmeans_segmentation(self, mask, weights, num_regions, spatial_weight=1.0, weight_scale=1.0, max_iter=300):
        """
        Perform K-means clustering with spatial and single-channel weight features.
        
        Parameters:
        - mask: Binary mask where 1 indicates the region of interest.
        - weights: A single-channel image (grayscale or weights) defining pixel-level features.
        - num_regions: Number of regions (clusters) to create.
        - spatial_weight: Weight factor for spatial coordinates.
        - weight_scale: Scale factor for the weights.
        - max_iter: Maximum iterations for K-means.

        Returns:
        - segmented_image: Image with regions visualized as distinct colors.
        - labels: Cluster labels for each pixel in the region.
        """
        # Get coordinates of the valid region
        y_coords, x_coords = np.where(mask > 0)
        valid_coords = np.column_stack((x_coords, y_coords))

        # Get weight values for valid region
        pixel_weights = weights[y_coords, x_coords]

        # Scale spatial and weight features
        spatial_features = valid_coords * spatial_weight
        weight_features = pixel_weights[:, np.newaxis] * weight_scale

        # Combine spatial and weight features
        features = np.hstack((spatial_features, weight_features))

        # Run K-means clustering
        kmeans = KMeans(n_clusters=num_regions, max_iter=max_iter, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # Create an output image with unique colors for each cluster
        segmented_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
        unique_colors = plt.cm.tab10(np.linspace(0, 1, num_regions))[:, :3] * 255  # Use Matplotlib's Tab10 colormap


        for i, (x, y) in enumerate(valid_coords):
            segmented_image[y, x] = unique_colors[labels[i]]

        return segmented_image, labels

    def distribute_points(self, labels, weights, mask, total_points):
        """
        Distribute points among regions proportionally based on region scores.

        Parameters:
        - labels: 1D array of region labels for valid pixels.
        - weights: Full 2D array of pixel weights (grayscale or other).
        - mask: Binary mask where 1 indicates the region of interest.
        - total_points: Total number of points to distribute.

        Returns:
        - points_per_region: List of number of points for each region.
        """
        # Extract valid pixel coordinates from the mask
        y_coords, x_coords = np.where(mask > 0)

        # Initialize scores for each region
        num_regions = labels.max() + 1  # Number of unique regions
        region_scores = np.zeros(num_regions)

        # Calculate scores for each region based on the weight values
        for region_id in range(num_regions):
            region_mask = (labels == region_id)  # Mask for current region
            region_weights = weights[y_coords[region_mask], x_coords[region_mask]]
            region_scores[region_id] = region_weights.sum()  # Sum of weights in the region

        # Normalize scores to distribute points proportionally
        total_score = region_scores.sum()
        points_per_region = (region_scores / total_score * total_points).round().astype(int)

        # Adjust to ensure the total matches exactly
        while points_per_region.sum() < total_points:
            points_per_region[np.argmax(region_scores)] += 1
        while points_per_region.sum() > total_points:
            points_per_region[np.argmax(points_per_region)] -= 1

        return points_per_region
    

    def generate_kmeans_points(self, mask, num_points, max_iter=300):
        """
        Generate equidistributed points on an arbitrary shape using K-means clustering.
        
        Parameters:
        - mask: A 2D numpy array (binary mask) where 1 indicates the shape and 0 is the background.
        - num_points: Number of points to generate.
        - max_iter: Maximum iterations for K-means clustering.
        
        Returns:
        - points: Array of shape (num_points, 2) with the coordinates of the points.
        """
        # Get coordinates of the valid region
        y_coords, x_coords = np.where(mask > 0)
        valid_coords = np.column_stack((x_coords, y_coords))

        # Run K-means clustering on the valid coordinates
        kmeans = KMeans(n_clusters=num_points, max_iter=max_iter, random_state=42, n_init=10)
        kmeans.fit(valid_coords)

        # Cluster centers are the resulting equidistributed points
        points = kmeans.cluster_centers_

        # Clip points to ensure they remain within the mask
        points = np.clip(points, [0, 0], [mask.shape[1] - 1, mask.shape[0] - 1])

        # Verify points are within the mask
        inside_mask = [mask[int(y), int(x)] > 0 for x, y in points]
        points = points[np.array(inside_mask)]

        return points


    def distribute_and_visualize_points(self, segmented_image, labels, points_per_region, mask):
        """
        Apply K-means to distribute points in each region and visualize them on the segmented image.

        Parameters:
        - segmented_image: Image with regions visualized as distinct colors.
        - labels: 1D array of region labels for valid pixels.
        - points_per_region: Number of points to distribute in each region.
        - mask: Binary mask (H x W) defining the valid region of interest.

        Returns:
        - combined_points: List of all (x, y) coordinates for distributed points.
        """
        num_regions = len(points_per_region)
        combined_points = []

        # Get valid pixel coordinates
        y_coords, x_coords = np.where(mask > 0)
        valid_coords = np.column_stack((x_coords, y_coords))

        for region_id, num_points in enumerate(points_per_region):
            if num_points > 0:
                # Create a binary mask for the current region
                region_mask = np.zeros_like(mask, dtype=np.uint8)
                region_pixel_indices = np.where(labels == region_id)
                region_pixels = valid_coords[region_pixel_indices]
                region_mask[region_pixels[:, 1], region_pixels[:, 0]] = 1

                # Generate points using K-means
                points = self.generate_kmeans_points(region_mask, num_points)
                combined_points.extend(points)

        segmented_image_ = segmented_image.copy()
        segmented_image_[mask == 0] = 255
        combined_points = np.array(combined_points)
        return combined_points, segmented_image_
    

    def get_points_smart_clustering(self, mask, weights):
        all_points = self.num_paths
        num_regions = 6  # Number of regions to divide  
        point_per_region = round(all_points / (2 * num_regions))
        remain_points = all_points - (point_per_region* num_regions)
        spatial_weight = 1.0  # Weight for spatial coordinates
        weight_scale = 0.5  # Scale for pixel weights (e.g., intensity)
        
        segmented_image, labels = self.weighted_kmeans_segmentation(mask, weights, num_regions, spatial_weight, weight_scale)
        total_points = remain_points  # Total points to distribute
        points_per_region = self.distribute_points(labels, weights, mask, total_points)
        final_points_per_region = points_per_region + 3
        combined_points, segmented_image_ = self.distribute_and_visualize_points(segmented_image, labels, final_points_per_region, mask)
        return combined_points, segmented_image_


    def set_attention_threshold_map(self):
        attn_map= torch.pow(self.attention_map, 2)
        attn_map_to_plot = (attn_map * self.mask) 
        weights = attn_map.numpy().astype(np.float32)
        
        mask= self.mask
        mask = (mask / mask.max()) * 255
        mask = mask.numpy().astype(np.uint8)

        self.inds, self.clustered_mask_to_plot = self.get_points_smart_clustering(mask, weights)

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 0] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 1] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()

        return attn_map_to_plot
        

    def get_attn(self):
        return self.attention_map
    
    def get_clustered_mask(self):
        return self.clustered_mask_to_plot

    def get_attn_map_to_plot(self):
        return self.attn_map_to_plot

    def get_inds(self):
        return self.inds

    def get_mask(self):
        return self.mask

   
class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        self.args = args

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(self.renderer.parameters(), lr=self.points_lr, betas=(0.9, 0.9), eps=1e-6)

      
    def zero_grad_(self):
        self.points_optim.zero_grad()
        
    def step_(self):
        self.points_optim.step()
     
    def get_lr(self):
        return self.points_optim.param_groups[0]['lr']
    


class AttnStoreProcessor(AttnProcessor2_0):
    def __init__(self, attn_maps):
        super().__init__()
        self.attn_maps = attn_maps

    def __call__(
            self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs
    ):
        batch_size, sequence_length, _ = hidden_states.shape

        # Ensure encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        # Standard attention computation
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute attention scores
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # **Store attention maps**
        if encoder_hidden_states is not hidden_states:
            # This is cross-attention
            self.attn_maps.append(attention_probs.detach().cpu())

        # Apply attention to values
        hidden_states = torch.bmm(attention_probs, value)

        # Reshape back to original dimensions
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
def get_net_attn_map(attn_maps, image_size, batch_size=2, instance_or_negative=False, detach=True):
    target_size = (image_size[0]//16, image_size[1]//16)
    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for attn_map in attn_maps:
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx] # (20, 32*32, 77) -> (10, 32*32, 77) # negative & positive CFG
        if len(attn_map.shape) == 4:
            attn_map = attn_map.squeeze()

        attn_map = upscale(attn_map, target_size) # (10,32*32,77) -> (77,64*64)
        net_attn_maps.append(attn_map) # (10,32*32,77) -> (77,64*64)

    net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)
    net_attn_maps = net_attn_maps.reshape(net_attn_maps.shape[0], 64,64) # (77,64*64) -> (77,64,64)

    return net_attn_maps
    

def inference_and_extract_attn(attn_maps, prompt, pipe, image, obj):
    net_attn_maps = get_net_attn_map(attn_maps, image.size)
    net_attn_maps = resize_net_attn_map(net_attn_maps, image.size)
    net_attn_maps = return_net_attn_map(net_attn_maps, pipe.tokenizer, prompt)

    # remove sos and eos
    net_attn_maps = [attn_map for attn_map in net_attn_maps if attn_map[1].split('_')[-1] != "<<|startoftext|>>"]
    net_attn_maps = [attn_map for attn_map in net_attn_maps if attn_map[1].split('_')[-1] != "<<|endoftext|>>"]
    ind = 4
    # for i, at_ in enumerate(net_attn_maps):
    #     if obj in at_[-1]:
    #         ind = i
    #         break
    attn = net_attn_maps[ind][0]
    attn = torch.tensor(np.array(attn))
    attn = (attn - attn.min()) / (attn.max() - attn.min())
    return attn

def register_attention_store(unet):
    attn_maps = []
    for name, module in unet.named_modules():
        if not name.split('.')[-1].startswith('attn2'):
            continue
        module.processor = AttnStoreProcessor(attn_maps)
    return unet, attn_maps



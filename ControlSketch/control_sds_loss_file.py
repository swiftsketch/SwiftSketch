import torch.nn as nn
import torch
import numpy as np
import wandb
from diffusers import StableDiffusionControlNetPipeline 
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import conditions_controlnet as cc


class ControlSDSLoss(nn.Module):
    def __init__(self, args, device):
        super(ControlSDSLoss, self).__init__()
        self.args = args
        self.device = device
        condition= self.args.condition
        self.conditioning_scale = self.args.conditioning_scale
        self.condition_image = self.create_condition_image(condition)
        controlnet = cc.controlnet(condition, self.device)
        self.resized_mask = []

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True).to(self.device)

        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        # creare caption
        if self.args.caption == "":
            self.create_caption()
        print("The text prompt for the controlnet sds loss:", self.args.caption, flush=True)

        # create text embeddings
        self.text_embeddings = None
        self.text_plus_uncond_embeddings = None
        self.embed_text()

    def create_condition_image(self, condition_name):  # create condition image for controlnet
        condition = cc.create_condition(self.args.input_image, condition_name)
        condition = self.creat_masked_condition(condition)
        condition.save(f"{self.args.output_dir}/{condition_name}_condition.png")
        if self.args.use_wandb:
            image_to_wandb = np.array(condition)
            wandb.log({f"{condition_name}_condition": wandb.Image(image_to_wandb)})
        final_condition = self.preprocessing_image_condtion(condition)
        return final_condition

    def creat_masked_condition(self, condition):
        im_np = np.array(condition)
        im_np = im_np / im_np.max()
        im_np = np.expand_dims(self.args.mask, axis=-1) * im_np
        im_np[self.args.mask < self.args.mask.mean()] = 0
        im_final = (im_np / im_np.max() * 255).astype(np.uint8)
        masked_im = Image.fromarray(im_final).resize((self.args.render_size, self.args.render_size))
        return masked_im
    
    def preprocessing_image_condtion(self, image):
        conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.args.render_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.args.render_size),
                transforms.ToTensor(),
            ]
        )
        image = image.convert("RGB")
        image = conditioning_image_transforms(image)
        image = image.to(dtype=torch.float16)
        image = image.to(device=self.device, dtype=torch.float16)
        image = image.unsqueeze(0)
        return image


    def create_caption(self):
        blip2processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        blip2model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                                   torch_dtype=torch.float16,
                                                                   resume_download=True).to(
            self.device)
        with torch.no_grad():
            inputs = blip2processor(self.args.input_image, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = blip2model.generate(**inputs, max_new_tokens=20)
            generated_text = blip2processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        caption = f"{generated_text}"
        self.args.caption = caption

        del blip2model
        del blip2processor
        torch.cuda.empty_cache()

    def embed_text(self):
        # tokenizer and embed text if using classifier free guidance                              
        text_input = self.pipe.tokenizer(self.args.caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                           max_length=text_input.input_ids.shape[-1],
                                           return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]

        self.text_embeddings = text_embeddings
        self.text_embeddings = self.text_embeddings.repeat_interleave(self.args.batch_size, 0)

        self.text_plus_uncond_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        self.text_plus_uncond_embeddings = self.text_plus_uncond_embeddings.repeat_interleave(self.args.batch_size, 0)

        del self.pipe.tokenizer
        del self.pipe.text_encoder
        torch.cuda.empty_cache()


    def forward(self, x):
        sds_loss = 0
        # encode rendered image
        x = x * 2. - 1.
        with torch.cuda.amp.autocast():
            init_latent_z = (self.pipe.vae.encode(x.to(dtype=torch.float16)).latent_dist.sample())
        latent_z = 0.18215 * init_latent_z  # scaling_factor * init_latents

        with torch.inference_mode():
            # sample timesteps
            timestep = torch.randint(
                low=50,
                high=min(950, self.args.diffusion_timesteps) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            eps = torch.randn_like(latent_z)
            # zt = alpha_t * latent_z + sigma_t * eps
            noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                    noised_latent_zt,
                    timestep,
                    encoder_hidden_states=self.text_embeddings,
                    controlnet_cond=self.condition_image,
                    conditioning_scale=self.conditioning_scale,
                    return_dict=False,
                )

                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]).to(dtype=torch.float16) for d in
                                          down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample]).to(
                    dtype=torch.float16)

                eps_t_uncond, eps_t = self.pipe.unet(
                    z_in,
                    timestep,
                    encoder_hidden_states=self.text_plus_uncond_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample.float().chunk(2)

            eps_t = eps_t_uncond + self.args.diffusion_guidance_scale * (eps_t - eps_t_uncond)

            # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
            grad_z = self.alphas[timestep] ** 0.5 * self.sigmas[timestep] * (eps_t - eps.float())
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

        sds_loss = grad_z.clone() * latent_z
        del grad_z

        sds_loss = sds_loss.sum(1).mean()
        return sds_loss 

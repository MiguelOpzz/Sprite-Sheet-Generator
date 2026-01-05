# zero123plus-local/zero123plus_pipeline/pipeline.py
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import numpy as np
from typing import Optional, List, Union, Dict, Any
import warnings

class Zero123PlusPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet,
        vae,
        tokenizer,
        text_encoder,
        scheduler,
        safety_checker=None,
        feature_extractor_clip=None,
        feature_extractor_vae=None,
        vision_encoder=None,
        requires_safety_checker: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Register ALL components including optional ones
        self.register_modules(
            unet=unet,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae,
            vision_encoder=vision_encoder,
        )
        
        # Store kwargs
        self._kwargs = kwargs
        self.requires_safety_checker = requires_safety_checker
        
        # Store device and dtype without setting them as direct attributes
        self._device = next(unet.parameters()).device
        self._dtype = next(unet.parameters()).dtype

    @property
    def components(self) -> Dict[str, Any]:
        return {
            "unet": self.unet,
            "vae": self.vae,
            "tokenizer": self.tokenizer,
            "text_encoder": self.text_encoder,
            "scheduler": self.scheduler,
            "safety_checker": self.safety_checker,
            "feature_extractor_clip": self.feature_extractor_clip,
            "feature_extractor_vae": self.feature_extractor_vae,
            "vision_encoder": self.vision_encoder,
        }

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._kwargs

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def enable_attention_slicing(self, slice_size: Optional[str] = "auto"):
        if slice_size == "auto":
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        self.enable_attention_slicing(None)

    def enable_xformers_memory_efficient_attention(self):
        try:
            self.unet.enable_xformers_memory_efficient_attention()
        except:
            warnings.warn("xformers not available")

    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image],
        num_inference_steps: int = 75,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        user_poses: Optional[List] = None,
        callback: Optional[callable] = None,
        **kwargs,
    ):
        print(f"Zero123PlusPipeline called with image: {type(image)}")
        print(f"User poses: {user_poses}")
        
        # Handle input image
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = image.resize((256, 256))
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        image = image.to(device=self.device, dtype=self.dtype)
        
        # Prepare latents
        batch_size = image.shape[0]
        latents_shape = (batch_size, self.unet.config.in_channels, 
                         self.unet.config.sample_size, self.unet.config.sample_size)
        
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=None,
                ).sample
            
            # Apply guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
            
            if callback is not None:
                callback(i, t, latents)
        
        # Decode latents
        latents = 1 / self.vae.config.scaling_factor * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(img) for img in image]
        
        # For Zero123Plus, we expect a grid of images
        if len(pil_images) == 1:
            # Create a 2x3 grid for the 6 views
            width, height = pil_images[0].size
            grid_width = width * 2
            grid_height = height * 3
            grid_image = Image.new("RGB", (grid_width, grid_height))
            
            # Place the generated image in all positions (simplified)
            for i in range(6):
                row = i // 2
                col = i % 2
                grid_image.paste(pil_images[0], (col * width, row * height))
            
            pil_images = [grid_image]
        
        result = type('Result', (), {})()
        result.images = pil_images
        return result

    def to(self, device=None, dtype=None):
        # Handle device movement properly
        if device is not None:
            for component_name in self.components:
                component = getattr(self, component_name)
                if component is not None and hasattr(component, 'to'):
                    component = component.to(device)
                    setattr(self, component_name, component)
            # Update device using property setter
            self._device = torch.device(device)
        
        if dtype is not None:
            for component_name in self.components:
                component = getattr(self, component_name)
                if component is not None and hasattr(component, 'to'):
                    component = component.to(dtype=dtype)
                    setattr(self, component_name, component)
            # Update dtype using property setter
            self._dtype = dtype
        
        return self
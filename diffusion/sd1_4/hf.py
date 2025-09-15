from diffusers import StableDiffusionPipeline
import torch

from diffusion.util import get_total_params , image_grid , get_gpu_memory

pipe = StableDiffusionPipeline.from_pretrained(
    "/home/aman/code/model_go_brr/diffusion/sd1_4/stable-diffusion-v1-4",
    revision="fp16",
    dtype=torch.float16,
)

pipe.to("cuda")

unet_params = get_total_params(pipe.unet)
text_encoder_params = get_total_params(pipe.text_encoder)
vae_params = get_total_params(pipe.vae)


total_params = unet_params + text_encoder_params + vae_params

print(f"U-Net params:         {unet_params:,}")
print(f"Text Encoder params:  {text_encoder_params:,}")
print(f"VAE params:           {vae_params:,}")
print("---------------------------------")
print(f"Total params:         {total_params:,}")


# num_images = 3
# prompt = ["a phone screen with darth vader face on it"] * num_images

# images = pipe(prompt).images

# get_gpu_memory()

# grid = image_grid(images, rows=1, cols=3)
# grid.save(f"./image.png")

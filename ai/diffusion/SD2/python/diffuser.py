from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

#model_id = "stabilityai/stable-diffusion-2-base"
model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

## GPU
#pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
#pipe = pipe.to("cuda")

# CPU
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float32)
pipe = pipe.to("cpu")

# Something
pipe.enable_attention_slicing()

# Set prompt
prompt = "a painting of a christmas scene in Camden, London UK with a lot of snow"

# Generate image
image = pipe(prompt).images[0]  

# Save image
image.save("/home/kampff/Downloads/camden_xmas.png")

#FIN
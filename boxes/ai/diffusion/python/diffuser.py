from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

#model_id = "stabilityai/stable-diffusion-2-base"
model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

prompt = "a blueprint style drawing of a school designed by buckminster fuller"
image = pipe(prompt).images[0]  
    
image.save("outputs/buckminster_school.png")

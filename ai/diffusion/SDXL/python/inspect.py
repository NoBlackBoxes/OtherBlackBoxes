from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# GPU
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipeline = pipeline.to("cuda")

# Something
pipeline.enable_attention_slicing()

# Set prompt
prompt = "very realistic painting of a nighttime christmas scene in London UK with a lot of snow"

# Set height and width based on UNet size and VAE scale factor
# - pipeline.unet.config.sample_size = 96
# - pipeline.vae_scale_factor = 8
height = 1024
width = 1024
callback_steps = 30

# Check inputs. Raise error if not correct
pipeline.check_inputs(prompt, callback_steps, '', height, width, 1)

# Define call parameters
batch_size = 1
device = pipeline._execution_device
guidance_scale = 9 # greater than 1 means do classifier free guidance
do_classifier_free_guidance = True

# Encode input prompt
num_images_per_prompt = 1
negative_prompt = None
text_embeddings = pipeline.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=do_classifier_free_guidance, negative_prompt=negative_prompt)

# Prepare timesteps
num_inference_steps = 50
pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = pipeline.scheduler.timesteps

# Prepare latent variables (random initialize)
generator = None
latents = None
num_channels_latents = pipeline.unet.in_channels
latents = pipeline.prepare_latents(
    batch_size * num_images_per_prompt,
    num_channels_latents,
    height,
    width,
    text_embeddings,
    device,
    generator,
    latents,
)

# Decode random latents
with torch.no_grad():
    image = pipeline.decode_latents(latents)


# Prepare extra step kwargs
eta = 0.0
extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

# Denoising loop
num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
with torch.no_grad():
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

        # Divert (after some iterations)
        if((i % 2) == 0):
            # predict the noise residual
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        else:
            # predict the noise residual
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=diversion_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            print(latents)

        # 8. Decode current latents
        image = pipeline.decode_latents(latents)

        # Save current image
        image = pipeline.numpy_to_pil(image)
        image[0].save("/home/kampff/Downloads/steps/{0}.png".format(i))

        # Report progress
        print("Iteration {0}...".format(i))

# Report Done
print("Done")

#FIN
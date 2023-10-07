from datetime import datetime
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch

# Check for GPU (also works for AMD GPUs using ROCm)
gpu_available = torch.cuda.is_available()
print("Is the GPU available? {0}".format(gpu_available))

# Set model ID
model_id = "stabilityai/stable-diffusion-2-1"

# Use the Euler scheduler here instead
scheduler =  EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# Load models (Large!)
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)

# Send to device
if(gpu_available):
    pipe = pipe.to("cuda")
else:
    pipe = pipe.to("cpu")

# Something
pipe.enable_attention_slicing()

# Set prompts
positive_prompt = 'old timey black and white photograph of a baby swiping on a smartphone'
negative_prompt = 'out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,'
#negative_prompt = ''

# Set height and width based on UNet size and VAE scale factor
# - pipe.unet.config.sample_size = 64 / 96
# - pipe.vae_scale_factor = 8
height = 512
width = 512
#height = 768
#width = 768

# Check inputs. Raise error if not correct
pipe.check_inputs(positive_prompt, height, width, 1)

# Define call parameters
batch_size = 1
device = pipe._execution_device
guidance_scale = 9.5 # greater than 1 means do classifier free guidance
do_classifier_free_guidance = True

# Encode input prompt
num_images_per_prompt = 1
text_embeddings = pipe._encode_prompt(positive_prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

# Prepare timesteps
num_inference_steps = 30
pipe.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = pipe.scheduler.timesteps

# Prepare latent variables (random initialize)
generator = None
latents = None
num_channels_latents = pipe.unet.in_channels
latents = pipe.prepare_latents(
    batch_size * num_images_per_prompt,
    num_channels_latents,
    height,
    width,
    text_embeddings.dtype,
    device,
    generator,
    latents,
)

# Prepare extra step kwargs
eta = 0.0
extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

# Denoising loop
num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
with torch.no_grad():
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        # Inputs:
        #  - latents (64x64 or 96x96), t (timestep), hidden_states ()
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
#        if False:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # UNet precicted noise in text and image embedding....the predicted noise is stepped by the difference between
            #   them scaled by the "guidance_scale" factor
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # Steps the latents...impements diffusion algorithm (subtract oredicited noise...basically...depends a bit on size of timestep...I guess to make it stable...)
            # - moves latemts through atent space towards the manifold with valid images...bu...guided by the text prompt...
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        else:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond
            # Just denoise initial random latents
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # Decode current latents
        image = pipe.decode_latents(latents)

        # Save current image
        image = pipe.numpy_to_pil(image)
        image[0].save("_tmp/output/steps/" + str(i).zfill(2) + '.png')

        # Report progress
        print("Iteration {0}...".format(i))

    # Decode latents
    image = pipe.decode_latents(latents)

    # Save final image
    name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    image = pipe.numpy_to_pil(image)
    image[0].save("_tmp/output/" + str(name) + '.png')

# Report Done
print("Done")

#FIN
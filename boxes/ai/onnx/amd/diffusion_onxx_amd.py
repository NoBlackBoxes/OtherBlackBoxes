from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnxruntime

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
text_encoder_path = '/home/kampff/Dropbox/Voight-Kampff/Technology/Models/stable-diffusion-2-1/text_encoder/model.onnx'
vae_decoder_path = '/home/kampff/Dropbox/Voight-Kampff/Technology/Models/stable-diffusion-2-1/vae_decoder/model.onnx'
unet_path = '/home/kampff/Dropbox/Voight-Kampff/Technology/Models/stable-diffusion-2-1/unet/model.onnx'

# Specify prompt
prompt = 'An amazing high resolution artistic photograph of a futuristic school building where all the teachers are robots'

# Specify execution providers
providers = [
    'ROCmExecutionProvider',
    'CPUExecutionProvider',
]

# Specify ONNS session options
so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

## Tokenize and encode prompt

# Convert prompt to tokens using CLIP Tokenizer...for now, just pick some token from the vocab file
positive_tokens = np.zeros((1,77), dtype=np.int32)
positive_tokens[0,0] = 49406
positive_tokens[0,1] = 1904
positive_tokens[0,2] = 1228
positive_tokens[0,3] = 49407
negative_tokens = np.zeros((1,77), dtype=np.int32)
negative_tokens[0,0] = 49406
negative_tokens[0,1] = 9250
negative_tokens[0,2] = 8159
negative_tokens[0,3] = 49407

# Load text encoder
session = onnxruntime.InferenceSession(text_encoder_path, sess_options=so, providers=providers)

# Show input name
session.get_inputs()[0].name

# Run model (Encode Text)
positive_text_embeddings = session.run([], {'input_ids':positive_tokens})[0]
negative_text_embeddings = session.run([], {'input_ids':negative_tokens})[0]

# Intermin cleanup
del session

## Generate noisy latents
height = 768
width = 768
latents = np.random.randn(1, 4, height // 8, width // 8).astype(np.float32)

## Create time steps
num_steps = 50
trained_steps = 1000
timesteps = np.linspace(0, trained_steps, num_steps, dtype=np.int64)[::-1]

## Denoise
guidance_scale = 9
step_scaling = 0.05

# Run model (predict noise)
#negative_text_embeddings = np.expand_dims(text_embeddings[0,:,:].detach().numpy(), 0)
#positive_text_embeddings = np.expand_dims(text_embeddings[1,:,:].detach().numpy(), 0)

# Open sessions
unet_session = onnxruntime.InferenceSession(unet_path, sess_options=so)
decoder_session = onnxruntime.InferenceSession(vae_decoder_path, sess_options=so)

for i in range(num_steps):
    print("Step {0}...".format(i))

    step = np.expand_dims(np.array(timesteps[i]), 0)
    negative_noise = unet_session.run([], {'sample':latents, 'timestep':step, 'encoder_hidden_states':negative_text_embeddings})[0]
    positive_noise = unet_session.run([], {'sample':latents, 'timestep':step, 'encoder_hidden_states':positive_text_embeddings})[0]
    predicted_noise = negative_noise + guidance_scale * (positive_noise - negative_noise)
    latents = latents - predicted_noise * step_scaling

    # Decode (why not?)
    latents_scaled = 1 / 0.18215 * latents # Why??
    output = decoder_session.run([], {'latent_sample':latents_scaled})[0]

    # Display interim image
    interim = np.copy(np.squeeze(output))
    interim = np.clip(interim / 2 + 0.5, 0, 1)
    interim = interim.transpose([1, 2, 0]) # transpose to Height*Width*Channel
    
    image = Image.fromarray(np.uint8(interim * 255)) 
    image.save("/home/kampff/Downloads/steps/{0}.png".format(i))

    #plt.figure()
    #plt.imshow(image)
    #plt.show()

# Intermin cleanup
del unet_session
del decoder_session

# FIN
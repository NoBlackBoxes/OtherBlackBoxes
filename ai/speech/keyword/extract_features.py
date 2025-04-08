# Extract MEL features from audio snippet
import os
import numpy as np
import wave
import torch
from python_speech_features import mfcc
from python_speech_features import logfbank
import matplotlib.pyplot as plt

# Locals libs
import dataset
import model

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/OtherBlackBoxes'
box_path = repo_path + '/ai/speech/keyword'
model_path = box_path + '/_tmp/custom.pt'

# Load example sound
wav_path = box_path + '/_tmp/dataset/five/0a2b400e_nohash_0.wav'
sound = dataset.load_wav(wav_path)
#plt.plot(sound)
#plt.show()

# Compute MFCCs
buffer = np.zeros((dataset.num_times, dataset.num_mfcc), dtype=np.float32)
mfccs = mfcc(sound, 
            samplerate=16000,
            winlen=0.025,
            winstep=0.010,
            numcep=dataset.num_mfcc,
            nfilt=40,
            nfft=512,
            lowfreq=300,
            highfreq=8000,
            appendEnergy=True,
            winfunc=np.hamming)
buffer[:mfccs.shape[0], :dataset.num_mfcc] = mfccs
mfccs = buffer.transpose()
#plt.imshow(mfccs)
#plt.show()

# Load model
custom_model = model.custom()
custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)
custom_model.eval()        # Put model in eval mode

# Prepare network input
input = torch.tensor(np.float32(mfccs))
input = torch.unsqueeze(torch.unsqueeze(input, 0), 0)

# Send to GPU
input = input.to(device)

# Inference
output = custom_model(input)

# Convert logits to probabilities
probs = torch.nn.functional.softmax(output, dim=1)

# Extract output
output = probs.cpu().detach().numpy()
output = np.squeeze(output)

# Report top class if confident
predicted_idx = np.argmax(output)
score = output[predicted_idx]
if score > 0.5 and predicted_idx != 0:
    print(f"DETECTED: {dataset.classes[predicted_idx]} : {score:.3f}")

#FIN
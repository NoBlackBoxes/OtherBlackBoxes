import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from python_speech_features import mfcc
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Locals libs
import model

# Reimport
import importlib
importlib.reload(model)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/repos/OtherBlackBoxes'
box_path = repo_path + '/boxes/ai/speech/keyword'
model_path = box_path + '/_tmp/sheila.pt'

# Set parameters
num_mfcc = 16
len_mfcc = 16

# Load model
custom_model = model.custom()
custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)

# List sound devices
Utilities.list_devices()

# Initiliaze microphone thread
microphone = Microphone.Microphone(input_device, num_channels, 'int32', sample_rate, buffer_size, max_samples)
microphone.gain = 10.0
microphone.start()

# Infer
try:
    while True:
        buffer = microphone.read_latest(16000)

        # Compute MFCCs
        mfccs = mfcc(buffer, 
                    samplerate=16000,
                    winlen=0.100,
                    winstep=0.064,
                    numcep=num_mfcc,
                    nfilt=num_mfcc,
                    nfft=4096,
                    preemph=0.0,
                    ceplifter=0,
                    appendEnergy=False,
                    winfunc=np.hanning)

        # Prepare network input
        input = torch.tensor(np.float32(mfccs.transpose()))
        input = torch.unsqueeze(torch.unsqueeze(input, 0), 0)

        # Send to GPU
        input = input.to(device)

        # Inference
        output = custom_model(input)

        # Extract output
        output = output.cpu().detach().numpy()
        output = np.squeeze(output)

        # Report
        print(output > 0.75)

finally:
    # Shutdown
    microphone.stop()

#FIN
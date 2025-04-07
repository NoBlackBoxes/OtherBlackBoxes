import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from python_speech_features import mfcc
import NB3.Sound.microphone as Microphone
import NB3.Sound.utilities as Utilities

# Locals libs
import dataset
import model

# Reimport
import importlib
importlib.reload(dataset)
importlib.reload(model)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/OtherBlackBoxes'
box_path = repo_path + '/ai/speech/keyword'
model_path = box_path + '/_tmp/custom.pt'

# Load model
custom_model = model.custom()
custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)
custom_model.eval()        # Put model in eval mode

# List sound devices
Utilities.list_devices()

# Initiliaze microphone thread
microphone = Microphone.Microphone(3, 1, 'int32', 48000, 4800, 48000*10)
microphone.gain = 1.0
microphone.start()

# Infer
try:
    while True:
        buffer = microphone.latest(48000)
        if len(buffer) != 48000:
            continue
        binned = buffer.reshape(-1, 3).mean(axis=1)
        #plt.plot(binned)
        #plt.show()

        # Compute MFCCs
        buffer = np.zeros((dataset.num_times, dataset.num_mfcc), dtype=np.float32)
        mfccs = mfcc(binned, 
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

        # Transpose MFCCs (rows = Fr, cols = time)
        mfccs = buffer.transpose()
        #plt.imshow(mfccs)
        #plt.show()

        # Prepare network input
        input = torch.tensor(np.float32(mfccs))
        input = torch.unsqueeze(torch.unsqueeze(input, 0), 0)

        # Send to GPU
        input = input.to(device)

        # Inference
        output = custom_model(input)

        # Extract output
        output = output.cpu().detach().numpy()
        output = np.squeeze(output)

        # Report
        score = np.max(output)
        #print(output)
        if score > 0.5:
            print(f"{dataset.detection_words[np.argmax(output)]} : {score}")

finally:
    # Shutdown
    microphone.stop()

#FIN
# Extract MEL features from audio snippet
import numpy as np
import wave
from python_speech_features import mfcc
from python_speech_features import logfbank
import matplotlib.pyplot as plt

# Set paths
root = '/home/kampff/NoBlackBoxes/OtherBlackBoxes/ai/speech/keyword'

# Set parameters
num_mfcc = 32

# Load example sound
wav_path = root + '/_tmp/dataset/five/0a2b400e_nohash_0.wav'
wav_obj = wave.open(wav_path)
num_channels = wav_obj.getnchannels()
sample_width = wav_obj.getsampwidth()
fs = wav_obj.getframerate()
num_frames = wav_obj.getnframes()
byte_data = wav_obj.readframes(num_frames)
sound = np.frombuffer(byte_data, dtype=np.int16)
wav_obj.close()

# Compute MFCCs
plt.plot(sound)
plt.show()
mfccs = mfcc(sound, 
            samplerate=fs,
            winlen=0.025,
            winstep=0.010,
            numcep=num_mfcc,
            nfilt=40,
            nfft=512,
            lowfreq=300,
            highfreq=8000,
            appendEnergy=True,
            winfunc=np.hamming)
mfccs = mfccs.transpose()
print(mfccs.shape)
plt.imshow(mfccs)
plt.show()

#FIN
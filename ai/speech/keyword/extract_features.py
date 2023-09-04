# Extract MEL features from audio snippet
import numpy as np
import wave
from python_speech_features import mfcc
from python_speech_features import logfbank

# Set paths
root = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes/boxes/ai/speech/keyword'

# Set parameters
num_mfcc = 16
len_mfcc = 16

# Load example sound
wav_path = root + '/_tmp/dataset/yes/0a7c2a8d_nohash_0.wav'
wav_obj = wave.open(wav_path)
num_channels = wav_obj.getnchannels()
sample_width = wav_obj.getsampwidth()
fs = wav_obj.getframerate()
num_frames = wav_obj.getnframes()
byte_data = wav_obj.readframes(num_frames)
sound = np.frombuffer(byte_data, dtype=np.int16)
wav_obj.close()

# Compute MFCCs
mfccs = mfcc(sound, 
            samplerate=fs,
            winlen=0.256,
            winstep=0.050,
            numcep=num_mfcc,
            nfilt=26,
            nfft=4096,
            preemph=0.0,
            ceplifter=0,
            appendEnergy=False,
            winfunc=np.hanning)




#FIN
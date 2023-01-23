import numpy as np
import cv2
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# Specify paths
video_path = '/home/kampff/Downloads/video.mp4'
audio_path = '/home/kampff/Downloads/sound.wav'

# Open video
video = cv2.VideoCapture(video_path)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(video.get(cv2.CAP_PROP_FPS))
length_video = num_frames / frame_rate

# Open sound
sample_rate, data = wavfile.read(audio_path)
num_samples = data.shape[0]
length_audio = num_samples / sample_rate
sound = data[:,0].asype(np.float32) / 32768.0

# Spectrogram
f, t, Sxx  = spectrogram(sound[:10*sample_rate], fs=sample_rate, window=('tukey', 0.25), nperseg=None, noverlap=None, nfft=sample_rate/10, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
print(Sxx.shape)
plt.pcolormesh(t, f, Sxx)
plt.imshow(np.log(Sxx))
plt.show()

# Create named window for diaply
cv2.namedWindow('preview')

# Process
for i in range(num_frames):

    # Read most recent frame
    ret, frame = video.read()

    # Display the resulting frame
    cv2.imshow('preview', frame)

    # Wait for a keypress, and quit if 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video handle
video.release()

# Destroy display window
cv2.destroyAllWindows()

#FIN
import os
import numpy as np
import torch
import pyaudio
import sound

# Locals libs
import model

# Reimport
import importlib
importlib.reload(model)
importlib.reload(sound)

# Initiliaze microphone thread
microphone = sound.microphone(4, 1600, pyaudio.paInt16, 16000, 10)
microphone.start()

# Infer
try:
    while True:
        buffer = microphone.read_latest(16000)
        avg = np.average(np.abs(buffer))
        print(avg)

finally:
    # Shutdown
    microphone.stop()

#FIN
import os
import time
import pyaudio
import numpy as np

# Locals libs
import libs.NB3_sound as sound

# Reimport
import importlib
importlib.reload(sound)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/repos/OtherBlackBoxes'
box_path = repo_path + '/boxes/ai/speech/chatNB3'

# Initiliaze microphone thread
sound.list_devices()

microphone = sound.microphone(4, 1600, pyaudio.paInt16, 16000, 10)
microphone.start()

# Wait a bit to let the microphone settle
time.sleep(1.5)

# --------------------------------------------------------------------------------
# Chat Loop
# --------------------------------------------------------------------------------
waiting = True
recording = False
try:
    while waiting:
        if microphone.is_speaking():
            recording = True
            waiting = False
            microphone.start_recording('speech.wav', 16000*5)
        
    while microphone.is_speaking() and microphone.is_recording():
        time.sleep(0.5)
        print('recording...')
        continue

    # Finish
    microphone.stop_recording()

finally:
    # Shutdown
    microphone.stop()
# FIN
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from threading import Thread

class stream:
    def __init__(self, stream_id=0):        
        # Configure audio recording
        CHUNK = 1600                # Buffer size
        FORMAT = pyaudio.paInt16    # Data type
        CHANNELS = 1                # Number of channels
        RATE = 16000                # Sample rate (Hz)
        RECORD_SECONDS = 3          # Duration

        # Get pyaudio object
        self.pya = pyaudio.PyAudio()

        # Open audio stream (from default device)
        self.stream = self.pya.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)        
        self.sound = np.zeros(CHANNELS * RATE * RECORD_SECONDS, dtype=np.float32)
        self.streaming = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        
    # method to start thread 
    def start(self):
        self.streaming = True
        self.thread.start()

    def update(self):
        while True :
            if self.streaming is False :
                break
            
            # Read raw data and append
            raw_data = self.stream.read(1600, exception_on_overflow = False)

            # Convert to numpy array
            integer_data = np.frombuffer(raw_data, dtype=np.int16)
            float_data = np.float32(integer_data)/32768.0

            # Concat and truncate
            self.sound = np.hstack([self.sound[1600:], float_data])
        
        # Shutdown
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()


    def read(self):
        return self.sound
    def stop(self):
        self.streaming = False
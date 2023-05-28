import numpy as np
import pyaudio
from threading import Thread

class stream:
    def __init__(self, buffer_size, format, num_channels, sample_rate, duration):        
        self.buffer_size = buffer_size
        self.format = format
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.duration = duration

        # Get pyaudio object
        self.pya = pyaudio.PyAudio()

        # Open audio stream (from default device)
        self.stream = self.pya.open(format=format, channels=num_channels, rate=sample_rate, input=True, frames_per_buffer=buffer_size)

        # Create rolling buffer
        self.sound = np.zeros(num_channels * sample_rate * duration, dtype=np.float32)
        self.streaming = False

        # Configure thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        
    # Start thread method
    def start(self):
        self.streaming = True
        self.thread.start()

    # Update thread method
    def update(self):
        while True :
            # End?
            if self.streaming is False :
                break
            
            # Read raw data and append
            raw_data = self.stream.read(self.buffer_size, exception_on_overflow = False)

            # Convert to numpy array
            integer_data = np.frombuffer(raw_data, dtype=np.int16)
            float_data = np.float32(integer_data)/32768.0

            # Concat and truncate
            self.sound = np.hstack([self.sound[1600:], float_data])
        
        # Shutdown thread
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()

    # Read sound method
    def read(self):
        return self.sound

    # Stop thread method
    def stop(self):
        self.streaming = False
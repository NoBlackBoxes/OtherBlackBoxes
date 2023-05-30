import numpy as np
import pyaudio
from threading import Thread

#
# Sound input thread (microphone)
#
class microphone:
    def __init__(self, buffer_size, format, sample_rate, max_duration):        
        self.buffer_size = buffer_size
        self.format = format
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.valid_samples = 0
        self.max_samples = 0

        # Get pyaudio object
        self.pya = pyaudio.PyAudio()

        # Open audio input stream (from default device)
        self.stream = self.pya.open(format=format, channels=1, rate=sample_rate, input=True, output=False, frames_per_buffer=buffer_size)

        # Create rolling buffer
        self.max_samples = sample_rate * max_duration
        self.sound = np.zeros(self.max_samples, dtype=np.float32)
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

            # Fill buffer...and then concat
            if self.valid_samples < self.max_samples:
                self.sound[self.valid_samples:(self.valid_samples + self.buffer_size)] = float_data
                self.valid_samples = self.valid_samples + self.buffer_size
            else:
                self.sound = np.hstack([self.sound[self.buffer_size:], float_data])
                self.valid_samples = self.max_samples
        
        # Shutdown thread
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()

    # Read sound method
    def read(self):
        num_valid_samples = self.valid_samples
        self.valid_samples = 0
        return self.sound[:num_valid_samples]

    # Reset sound input
    def reset(self):
        self.sound = np.zeros(self.max_samples, dtype=np.float32)
        self.valid_samples = 0
        return

    # Stop thread method
    def stop(self):
        self.streaming = False

#
# Sound output thread (speaker)
#
class speaker:
    def __init__(self, buffer_size, format, sample_rate):        
        self.buffer_size = buffer_size
        self.format = format
        self.sample_rate = sample_rate
        self.current_sample = 0
        self.max_samples = 0

        # Get pyaudio object
        self.pya = pyaudio.PyAudio()

        # Open audio output stream (from default device)
        self.stream = self.pya.open(format=format, channels=1, rate=sample_rate, input=False, output=True, frames_per_buffer=buffer_size)

        # Create rolling buffer
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
        empty_buffer = np.zeros(self.buffer_size, dtype=np.int16)
        while True :
            # End?
            if self.streaming is False :
                break
            
            # Playing?
            if self.current_sample < self.max_samples:
                # Write sound data buffer
                float_data = self.sound[self.current_sample:(self.current_sample + self.buffer_size)]
                integer_data = np.int16(float_data * 32768.0)
                self.stream.write(integer_data, self.buffer_size, exception_on_underflow = False)

                # Increment buffer position
                self.current_sample = self.current_sample + self.buffer_size
            else:
                self.stream.write(empty_buffer, self.buffer_size, exception_on_underflow = False)
        
        # Shutdown thread
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()

    # Write sound method
    def write(self, sound):
        num_samples = np.shape(sound)[0]
        max_samples = num_samples + (self.buffer_size - (num_samples % self.buffer_size))
        self.sound = np.zeros(max_samples)
        self.sound[:num_samples] = sound
        self.current_sample = 0
        self.max_samples = max_samples
        return

    # Reset sound output
    def reset(self):
        self.current_sample = 0
        return

    # Check if for sound output is finished
    def playing(self):
        if self.current_sample < self.max_samples:
            return True
        else:
            return False

    # Stop thread method
    def stop(self):
        self.streaming = False
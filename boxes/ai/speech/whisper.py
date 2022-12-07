import torch
import numpy as np
import pyaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Congure audio recording
CHUNK = 1600                # Buffer size
FORMAT = pyaudio.paInt16    # Data type
CHANNELS = 1                # Number of channels
RATE = 16000                # Sample rate (Hz)
RECORD_SECONDS = 5          # Duration

# Get pyaudio object
p = pyaudio.PyAudio()

# Open audio stream (from default device)
stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK) 

# Append frames of data
sound = np.zeros(CHANNELS * RATE * RECORD_SECONDS, dtype=np.float32)
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    # Read raw data and append
    raw_data = stream.read(CHUNK, exception_on_overflow = False)
    
    # Convert to numpy array
    integer_data = np.frombuffer(raw_data, dtype=np.int16)
    float_data = np.float32(integer_data)/32768.0

    # Concat
    sound[(i *CHUNK):((i *CHUNK) + CHUNK)] = float_data

    # Report volume (on left)
    print("Volume: {0:.2f}".format(np.mean(np.abs(float_data))))

# Shutdown
stream.stop_stream()
stream.close()
p.terminate()

# Extract features
inputs = processor(sound, sampling_rate=RATE, return_tensors="pt")
input_features = inputs.input_features

# Generate IDs
generated_ids = model.generate(inputs=input_features)

# Transcribe
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)

#FIN
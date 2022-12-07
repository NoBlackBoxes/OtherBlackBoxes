import torch
import pyaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Congure audio recording
CHUNK = 4800                # Buffer size
FORMAT = pyaudio.paInt32    # Data type
CHANNELS = 2                # Number of channels
RATE = 48000                # Sample rate (Hz)
RECORD_SECONDS = 500        # Duration

# Get pyaudio object
p = pyaudio.PyAudio()

# Open audio stream (from default device)
stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK) 

# Append frames of data
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    # Read raw data and append
    raw_data = stream.read(CHUNK)
    frames.append(raw_data)
    
    # Convert to numpy array
    interleaved_data = np.frombuffer(raw_data, dtype=np.int16)

    # Extract left and right values
    left = interleaved_data[::2] 
    right = interleaved_data[1::2]  

    # DO SOME PROCESSING HERE #

    # Report volume (on left)
    print("L: {0:.2f}, R: {1:.2f}".format(np.mean(np.abs(left)), np.mean(np.abs(right))))

# Shutdown
stream.stop_stream()
stream.close()
p.terminate()


# Extract features
inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
input_features = inputs.input_features

# Generate IDs
generated_ids = model.generate(inputs=input_features)

# Transcribe
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)

#FIN
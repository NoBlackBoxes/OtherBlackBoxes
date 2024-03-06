import os
import numpy as np
import moviepy.editor as mp
import scipy.signal as signal
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import wave

# Extract audio from video
video_path = "_tmp/LBB_electronics_2_2021.mkv"
video = mp.VideoFileClip(video_path)
width = video.size[0]
height = video.size[1]
fps = video.fps
audio = video.audio 
sample_rate = audio.fps
if sample_rate != 16000:
    audio.write_audiofile("_tmp/original.wav")
    os.system("ffmpeg -y -i _tmp/original.wav -ac 1 -ar 16000 _tmp/audio.wav")
else:
    audio.write_audiofile("_tmp/audio.wav")

# Specify video path AND OTHER PARAMETERS
wav_path = "_tmp/audio.wav"
sample_rate = 16000
max_chunk_size = sample_rate * 30
buffer_size = sample_rate // 10
avg_pools = 5
sr_threshold = 0.2
freq_bins = np.fft.fftfreq(buffer_size, 1.0/sample_rate)[1:]
output_path = "_tmp/transcript.csv"

# ----------
def find_silence(audio):
    speech_ratios = []
    num_buffers = audio.size // buffer_size
    current_buffer = 0
    rolling_average = 0.0
    speech_detected = False
    silence_buffer = num_buffers
    while current_buffer < num_buffers:
        offset = (current_buffer * buffer_size)
        buffer = audio[offset:(offset + buffer_size)]
        amplitudes = np.abs(np.fft.fft(buffer))[1:]
        energies = amplitudes**2

        # Compute total energy
        energy_per_freq = {}
        for (i, freq) in enumerate(freq_bins):
            if abs(freq) not in energy_per_freq:
                energy_per_freq[abs(freq)] = energies[i] * 2
        total_energy = sum(energy_per_freq.values())

        # Compute voice energy
        voice_energy = 0
        for f in energy_per_freq.keys():
            if 300 < f < 3000:                      # Human voice range
                voice_energy += energy_per_freq[f]

        # Compute speech ratio
        speech_ratio = voice_energy/total_energy

        # Update average
        rolling_average = ((rolling_average * (avg_pools - 1)) + speech_ratio) / avg_pools

        # Check for speech
        if rolling_average > 0.75:
            speech_detected =  True

        # Check for silence
        if speech_detected and (rolling_average < 0.2):
            silence_buffer = current_buffer
            break

        # Next
        current_buffer += 1

    return silence_buffer * buffer_size
# ----------

# Load speech recognition processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Load WAV file
wav_file = wave.open(wav_path, 'rb')

# Open output file
output_file = open(output_path, 'w')

# Transcribe
read_position = 0
while True:

    # Extract and convert next audio chunk
    raw_data = wav_file.readframes(max_chunk_size)
    if(len(raw_data) < max_chunk_size):
        print(len(raw_data))
        break
    integer_data = np.frombuffer(raw_data, dtype=np.int16)
    float_data = np.float32(integer_data)/32768.0

    # Find period of silence
    silence_offset = find_silence(float_data)

    # Set file position back for next read
    start_position = read_position
    read_position = read_position + silence_offset
    wav_file.setpos(read_position)

    # Extract features
    inputs = processor(float_data[:silence_offset], sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features

    # Generate IDs
    generated_ids = model.generate(inputs=input_features, max_new_tokens=1024)

    # Transcribe
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Print transcription
    print(transcription + ' - ({0}::{1})'.format(start_position, read_position))

    # Write transcription
    output_file.write(transcription + '::{0}::{1}\n'.format(start_position, read_position))

# Shutdown
wav_file.close()
output_file.close()

# FIN

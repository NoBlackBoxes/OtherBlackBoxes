import os
import numpy as np
import torch
import wave
import random
import matplotlib.pyplot as plt
from python_speech_features import mfcc

# Set parameters
sample_rate = 16000
num_mfcc = 32
num_times = 99
silence_reduction_factor = 0.01
noise_addition_factor = 0.5

# Specify words
non_words = ["silence", "noise"]
command_words = ["backward", "down", "eight", "five", "follow", "forward", "four", "go", "learn", "left", "nine", "no", "off", "on", "one", "right", "seven", "six", "stop", "three", "two", "up", "visual", "yes", "zero"]
distraction_words = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
classes = non_words + command_words + distraction_words

# Define dataset class (which extends the utils.data.Dataset module)
class custom(torch.utils.data.Dataset):
    def __init__(self, wav_paths, targets, noise, augment=False):
        self.wav_paths = wav_paths
        self.targets = targets
        self.noise = noise
        self.augment = augment
        self.mel_matrix = generate_mel_matrix()

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        target = self.targets[idx]

        # Load Sound (from WAV file or Noise/Silence)
        if target == 0: # This is a "silence" example
            start_frame = random.randint(0, len(self.noise)-sample_rate)
            sound = self.noise[start_frame:(start_frame+sample_rate)] * silence_reduction_factor # Silence is attenuated "noise"
        elif target == 1: # This is a "noise" example
            start_frame = random.randint(0, len(self.noise)-sample_rate)
            sound = self.noise[start_frame:(start_frame+sample_rate)]
        else:
            sound = load_wav(wav_path)
            if len(sound) < sample_rate:
                # Pad short WAV files
                buffer = np.zeros(sample_rate-len(sound))
                sound = np.concatenate([sound, buffer])
            # Augment?
            if self.augment:
                start_frame = random.randint(0, len(self.noise)-sample_rate)
                noise = self.noise[start_frame:(start_frame+sample_rate)]
                noise_multiplier = random.uniform(0.0, noise_addition_factor)
                sound = sound + (noise_multiplier * noise)

        # Compute Features
        features = process_sound(sound, mel_matrix=self.mel_matrix)
        
        # Add channel dimension
        features = np.expand_dims(features, 0)

        # Convert to Float32 (input) and Long (target)
        features = np.float32(features)
        target = np.long(target)

        return features, target

# Load WAV
def load_wav(path):
        wav_obj = wave.open(path)
        num_frames = wav_obj.getnframes()
        byte_data = wav_obj.readframes(num_frames)
        sound = np.frombuffer(byte_data, dtype=np.int16)
        wav_obj.close()
        sound_f = sound.astype(np.float32) / 32768.0
        return sound_f

# Generate mel matrix
def generate_mel_matrix():
    sample_rate = 16000
    mel_fft_length = 512
    mel_num_bins = 32

    mel_matrix = np.zeros((mel_fft_length // 2 + 1, mel_num_bins))
    freq_bins = np.linspace(0, sample_rate / 2, mel_fft_length // 2 + 1)
    freq_bins_mel = 1127.0 * np.log(1.0 + freq_bins / 700.0)
    mel_bins = np.linspace(1127.0 * np.log(1.0 + 60 / 700.0), 1127.0 * np.log(1.0 + 3800 / 700.0), mel_num_bins + 2)

    for i in range(mel_num_bins):
        lower = mel_bins[i]
        center = mel_bins[i + 1]
        upper = mel_bins[i + 2]
        mel_matrix[:, i] = np.maximum(0, np.minimum((freq_bins_mel - lower) / (center - lower), (upper - freq_bins_mel) / (upper - center)))

    return mel_matrix

# Process sound
def process_sound(sound, mel_matrix=None):
    # Assumes 16000 samples at 16 kHz (1 second) of audio (1 channel)
    # Float32, -1.0 to 1.0
    num_samples = 16000
    sample_rate = 16000

    # Parameters
    mel_window_length_samples = 400     # 25 ms
    mel_hop_length_samples = 160        # 10 ms
    mel_fft_length = 512
    if mel_matrix is None:
        mel_matrix = generate_mel_matrix()

    # Compute spectrogram
    frames = []
    for i in range(0, 16000 - mel_window_length_samples + 1, mel_hop_length_samples):
        frame = sound[i:i+mel_window_length_samples]
        windowed = frame * np.hanning(mel_window_length_samples)
        frames.append(np.abs(np.fft.rfft(windowed, mel_fft_length)))
    spectrogram = np.stack(frames)

    # Apply mel filters and take log
    mel_spectrogram = np.dot(spectrogram, mel_matrix)
    #log_mel_spectrogram = np.log(mel_spectrogram + 0.001)

    # Normalise
    mel_spectrogram -= np.mean(mel_spectrogram, axis=0, keepdims=True)
    mel_spectrogram /= (3 * np.std(mel_spectrogram, axis=0, keepdims=True))

    return mel_spectrogram.T


# Load dataset
def prepare(dataset_folder, split):

    # Find all WAV folders
    wav_folders = []
    for f in os.listdir(dataset_folder):
        if os.path.isdir(dataset_folder + '/' + f):
            if f != '_background_noise_':
                wav_folders.append(dataset_folder + '/' + f)

    # Find all WAV files
    wav_paths = []
    targets = []
    for f in wav_folders:
        paths = os.listdir(f)
        full_paths = []
        for path in paths:
            full_paths.append(f + '/' + path)
        num_paths = len(full_paths)
        wav_paths.extend(full_paths)
        targets.extend([os.path.basename(f)] * num_paths) # replicate this target label and append

    # Load all Noise files
    noise_arrays = []
    for f in os.listdir(f"{dataset_folder}/_background_noise_"):
        if f.endswith("wav"):
            noise_path = f"{dataset_folder}/_background_noise_/{f}"
            sound = load_wav(noise_path)
            noise_arrays.append(sound)
    noise_data = np.concatenate(noise_arrays)

    # Include samples for "Silence"
    num_random = int(len(wav_paths) / len(command_words))
    for i in range(num_random):
        wav_paths.append("silence")
        targets.append("silence")

    # Include samples for "Noise"
    num_random = int(len(wav_paths) / len(command_words))
    for i in range(num_random):
        wav_paths.append("noise")
        targets.append("noise")

    # Determine targets
    target_list = []
    for t in targets:
        target_index = classes.index(t)
        target_list.append(target_index)

    # Convert to arrays
    wav_paths = np.array(wav_paths)
    target_array = np.array(target_list)

    # Split train/test
    num_samples = len(targets)
    num_train = int(num_samples * split)
    num_test = num_samples - num_train
    indices = np.arange(num_samples)
    shuffled = np.random.permutation(indices)
    train_indices = shuffled[:num_train]
    test_indices = shuffled[num_train:]

    # Bundle
    train_data = (wav_paths[train_indices], target_array[train_indices])
    test_data = (wav_paths[test_indices], target_array[test_indices])

    return train_data, test_data, noise_data

#FIN
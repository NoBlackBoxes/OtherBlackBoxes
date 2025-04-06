import os
import numpy as np
import torch
import wave
import random
from python_speech_features import mfcc

# Set parameters
num_mfcc = 32
num_times = 99

# Specify words
non_word = ["noise"]
command_words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
distraction_words = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
detection_words = non_word + command_words + distraction_words

# Define dataset class (which extends the utils.data.Dataset module)
class custom(torch.utils.data.Dataset):
    def __init__(self, wav_paths, targets, noise, transform=None, target_transform=None, augment=False):
        self.wav_paths = wav_paths
        self.targets = targets
        self.noise = noise
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        target = self.targets[idx]

        # Load WAV        
        if target[0] == 1.0: # This is a "noise" example
            start_frame = random.randint(0, len(self.noise)-16000)
            sound = self.noise[start_frame:(start_frame+16000)]
        else:
            sound = load_wav(wav_path)
            if len(sound) < 16000:
                buffer = np.zeros(16000-len(sound))
                sound = np.concatenate([sound, buffer])
            # Augment?
            if self.augment:
                start_frame = random.randint(0, len(self.noise)-16000)
                noise = self.noise[start_frame:(start_frame+16000)]
                sound = sound + (0.5 * noise)

        # Compute MFCCs
        buffer = np.zeros((num_times, num_mfcc), dtype=np.float32)
        mfccs = mfcc(sound, 
                    samplerate=16000,
                    winlen=0.025,
                    winstep=0.010,
                    numcep=num_mfcc,
                    nfilt=40,
                    nfft=512,
                    lowfreq=300,
                    highfreq=8000,
                    appendEnergy=True,
                    winfunc=np.hamming)

        # Fill buffer
        buffer[:mfccs.shape[0], :num_mfcc] = mfccs

        # Transpose MFCCs (rows = Fr, cols = time)
        mfccs = buffer.transpose()
        
        # Add channel dimension
        mfccs = np.expand_dims(mfccs, 0)

        # Convert to Float32
        mfccs = np.float32(mfccs)
        target = np.float32(target)

        return mfccs, target

# Load WAV
def load_wav(path):
        wav_obj = wave.open(path)
        num_frames = wav_obj.getnframes()
        byte_data = wav_obj.readframes(num_frames)
        sound = np.frombuffer(byte_data, dtype=np.int16)
        wav_obj.close()
        sound_f = sound.astype(np.float32) / 32768.0
        return sound_f

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

    # Include placeholders for "Noise"
    num_random = len(wav_paths)
    for i in range(num_random):
        wav_paths.append("noise")
        targets.append("noise")

    # Determine target
    target_lists = []
    for t in targets:
        target_word = t if t in detection_words else "noise"
        target_list = [1.0 if word == target_word else 0.0 for word in detection_words]
        target_lists.append(target_list)

    # Convert to arrays
    wav_paths = np.array(wav_paths)
    target_array = np.array(target_lists)

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
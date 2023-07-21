import os
import numpy as np
import torch
import wave
from python_speech_features import mfcc

# Set parameters
num_mfcc = 32
len_mfcc = 37

# Define dataset class (which extends the utils.data.Dataset module)
class custom(torch.utils.data.Dataset):
    def __init__(self, wav_paths, targets, num_classes, transform=None, target_transform=None, augment=False):
        self.wav_paths = wav_paths
        self.targets = targets
        self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        target = self.targets[idx]

        # Load WAV
        wav_obj = wave.open(wav_path)
        fs = wav_obj.getframerate()
        num_frames = wav_obj.getnframes()
        byte_data = wav_obj.readframes(num_frames)
        sound = np.frombuffer(byte_data, dtype=np.int16)
        wav_obj.close()

        # Compute MFCCs
        buffer = np.zeros((len_mfcc, num_mfcc), dtype=np.float64)
        mfccs = mfcc(sound, 
                    samplerate=fs,
                    winlen=0.100,
                    winstep=0.025,
                    numcep=num_mfcc,
                    nfilt=num_mfcc,
                    nfft=4096,
                    preemph=0.0,
                    ceplifter=0,
                    appendEnergy=False,
                    winfunc=np.hanning)

        # Fill buffer
        buffer[:mfccs.shape[0], :num_mfcc] = mfccs

        # Transpose MFCCs (rows = Fr, cols = time)
        mfccs = buffer.transpose()

        # Augment?
        if self.augment:
            mfccs = augment(mfccs)
        
        # Add channel dimesnion
        mfccs = np.expand_dims(mfccs, 0)

        # COnvert to FLoat32
        mfccs = np.float32(mfccs)
        target = np.float32(target)

        return mfccs, target

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
    class_id = 0
    for f in wav_folders:
        paths = os.listdir(f)
        full_paths = []
        for path in paths:
            full_paths.append(f + '/' + path)
        num_paths = len(full_paths)
        wav_paths.extend(full_paths)
        targets.extend([class_id] * num_paths) # replicate this target label and append
        class_id += 1 

    # Build target array
    num_classes = class_id
    target_arrays = []
    base_target = np.zeros(num_classes)
    for t in targets:
        t_array = np.copy(base_target)
        t_array[t] = 1.0
        target_arrays.append(t_array)

    # Convert to arrays
    wav_paths = np.array(wav_paths)
    target_arrays = np.array(target_arrays)

    # Split train/test
    num_samples = len(targets)
    num_train = int(num_samples * split)
    num_test = num_samples - num_train
    indices = np.arange(num_samples)
    shuffled = np.random.permutation(indices)
    train_indices = shuffled[:num_train]
    test_indices = shuffled[num_train:]

    # Bundle
    train_data = (wav_paths[train_indices], target_arrays[train_indices])
    test_data = (wav_paths[test_indices], target_arrays[test_indices])

    return train_data, test_data, num_classes

# Augment
def augment(mfccs):

    # Augment MFCCs
    
    return mfccs

#FIN
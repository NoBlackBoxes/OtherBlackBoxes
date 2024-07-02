import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import glob
import training.utilities as utilities

# Define dataset class (which extends the utils.data.Dataset module)
class dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, augment=False):
        self.image_paths = image_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])

        # Augment (or just resize)
        if self.augment:
            image = augment(image)
        else:
            image = cv2.resize(image, (64,64))

        # Convert to scaled tensor
        image = image.transpose(2, 0, 1)
        input = torch.tensor(image, dtype=torch.float32) / 255.0
        return input, input

# Load dataset
def prepare(dataset_folder, split):

    # Filter train and test datasets
    image_paths = filter(dataset_folder)

    # Split train/test
    num_samples = len(image_paths)
    num_train = int(num_samples * split)
    num_test = num_samples - num_train
    indices = np.arange(num_samples)
    shuffled = np.random.permutation(indices)
    train_indices = shuffled[:num_train]
    test_indices = shuffled[num_train:]

    # Bundle
    train_data = image_paths[train_indices]
    test_data = image_paths[test_indices]

    return train_data, test_data

# Filter dataset
def filter(dataset_folder):
    image_paths = glob.glob(dataset_folder + "/*aligned.jpg")
    # Check that this image is acceptable?
    return np.array(image_paths)

# Augment
def augment(image):
    rotation_range = 10
    zoom_range = 0.05
    shift_range = 0.05
    random_flip = 0.4
    augmented_image = utilities.random_transform(image, rotation_range, zoom_range, shift_range, random_flip)
    augmented_image = cv2.resize(augmented_image, (64,64))
    return augmented_image

#FIN
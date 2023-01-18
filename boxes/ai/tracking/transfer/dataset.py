import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Define dataset class (which extends the utils.data.Dataset module)
class custom(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets, transform=None, target_transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, target

#FIN
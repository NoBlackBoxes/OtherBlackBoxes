import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

# Get user name
username = os.getlogin()

# Define dataset class (which extends the utils.data.Dataset module)
class custom(torch.utils.data.Dataset):
    def __init__(self, num_fakes, transform=None, target_transform=None, augment=False):
        self.num_fakes = num_fakes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_fakes

    def __getitem__(self, idx):
        image = np.zeros((224,224,3), dtype=np.uint8)
        x = np.random.randint(0,224)
        y = np.random.randint(0,224)
        image[y,x] = 255
        x = np.random.randint(0,224)
        y = np.random.randint(0,224)
        target = np.array([x / 224.0, y /224.0])

        # Generate heatmap
        heatmap = generate_heatmap(target)

        # Set transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, heatmap

# Generate heatmap
def generate_heatmap(target):
    heatmap = np.zeros((224,224), dtype=np.float32)
    x = target[0]
    y = target[1]
    ix = int(np.floor(x * 224))
    iy = int(np.floor(y * 224)) 
    heatmap[iy][ix] = 1.0
    heatmap = cv2.GaussianBlur(heatmap, ksize=(51,51), sigmaX=9, sigmaY=9)
    heatmap = cv2.resize(heatmap, (14,14), interpolation=cv2.INTER_LINEAR)
    heatmap = heatmap / np.sum(heatmap[:])
    heatmap = np.expand_dims(heatmap, axis=0)

    return heatmap

#FIN
#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')

# Set library paths
import sys
sys.path.append(libs_path)
#----------------------------------------------------------

# Import libraries
import torch
from torchsummary import summary
import numpy as np
import glob
import cv2
import training.model as model
import training.dataset as dataset
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches
from importlib import reload

# Debug
debug = True

# Specify paths
box_path = base_path
model_path = box_path + '/_tmp/models/interim/training_100.pth'
dataset_folder_A = base_path + '/_tmp/dataset/B'
dataset_folder_B = base_path + '/_tmp/dataset/C'

# Prepare datasets
reload(dataset)
train_data_A, test_data_A = dataset.prepare(dataset_folder_A, 0.8)
train_data_B, test_data_B = dataset.prepare(dataset_folder_B, 0.8)

# Create datasets
train_dataset_A = dataset.dataset(image_paths=train_data_A, augment=False)
test_dataset_A = dataset.dataset(image_paths=test_data_A, augment=False)
train_dataset_B = dataset.dataset(image_paths=train_data_B, augment=False)
test_dataset_B = dataset.dataset(image_paths=test_data_B, augment=False)

# Create data loaders
train_dataloader_A = torch.utils.data.DataLoader(train_dataset_A, batch_size=32, shuffle=True)
test_dataloader_A = torch.utils.data.DataLoader(test_dataset_A, batch_size=32, shuffle=True)
train_dataloader_B = torch.utils.data.DataLoader(train_dataset_B, batch_size=32, shuffle=True)
test_dataloader_B = torch.utils.data.DataLoader(test_dataset_B, batch_size=32, shuffle=True)

# Inspect dataset?
if debug:
    train_features, train_targets = next(iter(train_dataloader_A))
    for i in range(9):
        plt.subplot(3,3,i+1)
        feature = train_features[i]
        image = np.transpose(feature, (1,2,0))
        image = np.uint8(image * 255.0)
        plt.imshow(image[:,:,::-1])
    plt.show()

# Load model
reload(model)
training_model = model.model()
training_model.load_state_dict(torch.load(model_path))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
training_model.to(device)

# Display image input and output
features, targets = next(iter(test_dataloader_B))
print(f"Feature batch shape: {features.size()}")
print(f"Targets batch shape: {targets.size()}")

# Let's run it
features_gpu = features.to(device)
outputs = training_model(features_gpu, 'A')
outputs = outputs.cpu().detach().numpy()

# Examine predictions
for i in range(9):
    plt.subplot(3,3,i+1)
    feature = features[i]
    image = np.transpose(feature, (1,2,0))
    image = np.uint8(image * 255.0)
    output = outputs[i]
    output = np.transpose(output, (1,2,0))
    output = np.uint8(output * 255.0)
    display = np.hstack((image,output))
    plt.imshow(display[:,:,::-1])
plt.show()

#FIN
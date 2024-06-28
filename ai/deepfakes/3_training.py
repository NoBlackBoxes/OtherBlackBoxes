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
#import training.utilities as utilities
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches
from importlib import reload

# Debug
debug = True

# Specify paths
box_path = base_path
model_path = box_path + '/_tmp/models/training.pth'
dataset_folder = base_path + '/_tmp/dataset'

# Load model
reload(model)
training_model = model.model()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
training_model.to(device)
summary(training_model, (3, 64, 64))
#summary(training_model, (512, 8, 8))

# Test input image
image_path = "/home/kampff/NoBlackBoxes/OtherBlackBoxes/ai/deepfakes/_tmp/dataset/B/beast_clips_3_aligned.jpg"
image = cv2.imread(image_path)
resized = cv2.resize(image, (64, 64))
transposed = resized.transpose(2, 0, 1)
image = np.expand_dims(transposed, 0)

# Preprocess image
input = torch.tensor(image, dtype=torch.float32) / 255.0

# Send to GPU
input = input.to(device)

# Inference
output = training_model(input, select='A')

#FIN
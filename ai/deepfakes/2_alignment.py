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
import numpy as np
import alignment.model as model
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches
from torchvision import transforms

# Debug
debug = True

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/OtherBlackBoxes'
box_path = repo_path + '/ai/deepfakes'
model_path = box_path + '/_tmp/models/alignment.pth'

# Load model
alignment_model = model.model()
alignment_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
alignment_model.to(device)

# Load test image
image_path = repo_path + '/ai/tracking/_data/people.jpg'
image = Image.open(image_path)
image = image.resize((256,256))
image = np.array(image)
if debug:
    original = np.copy(image)
    plt.imshow(original)
    plt.show()
image = image.transpose(2, 0, 1)
image = np.expand_dims(image, 0)

# Detect Face(s)
with torch.no_grad():

    # Move model to device
    alignment_model.to(device)

    # Preprocess image
    input = torch.tensor(image, dtype=torch.float32)

    # Send to GPU
    input = input.to(device)
    input = input.flip(-3)  # RGB to BGR
    input = input - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)

    # Inference
    output = alignment_model(input)

    # Parse output into keypoints

    # Align face

#FIN
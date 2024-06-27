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
import alignment.utilities as utilities
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches

# Debug
debug = True

# Specify paths
box_path = base_path
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
image_path = base_path + '/_data/people.jpg'
image = Image.open(image_path)
image = np.array(image)
original = np.copy(image)
B = np.copy(image[:,:,2])
R = np.copy(image[:,:,0])
image[:,:,0] = B
image[:,:,2] = R

# Load bounding box coordinates
file_path = image_path[:-4] + '.txt'
bbox = np.genfromtxt(file_path, delimiter=",")

# 
left = bbox[0]


# CROP
# Resize to 256,256

#crop_ratio=0.55
#centre = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
#face_size = ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 2.0
#enlarged_face_box_size = (face_size / crop_ratio)
if debug:
    plt.imshow(original)
    plt.show()
image = image.transpose(2, 0, 1)
image = np.expand_dims(image, 0)

# Detect Face Keypoints (68)
with torch.no_grad():

    # Move model to device
    alignment_model.to(device)

    # Preprocess image
    input = torch.tensor(image, dtype=torch.float32) / 255.0

    # Send to GPU
    input = input.to(device)

    # Inference
    heatmaps, stem_feats, hg_feats = alignment_model(input)
    
    # Post-process keypoints
    landmarks, landmark_scores = utilities.decode(heatmaps)    
    landmark = landmarks[0]
    hh, hw = heatmaps.size(2), heatmaps.size(3)
    left = 0
    right = 255
    top = 0
    bottom = 255
    landmark[:, 0] = landmark[:, 0] * (right - left) / hw + left
    landmark[:, 1] = landmark[:, 1] * (bottom - top) / hh + top

    if debug:
        plt.imshow(original)
        plt.plot(landmark[:,0], landmark[:,1], 'y.')
        plt.show()

    # Align face

#FIN
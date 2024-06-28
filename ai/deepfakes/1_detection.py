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
import glob
import cv2
import detection.model as model
import detection.utilities as utilities
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches

# Debug
debug = True

# Specify paths
box_path = base_path
model_path = box_path + '/_tmp/models/detection.pth'
input_folder = base_path + '/_tmp/dataset/B'

# Load model
detection_model = model.model()
detection_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Select CPU or GPU device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
detection_model.to(device)

# Find all input files
image_paths = glob.glob(input_folder + "/*.jpg")

# Detect face bounding boxes in all images
for i, image_path in enumerate(image_paths):

    # Load test image
    image = cv2.imread(image_path)
    image = np.array(image)
    if debug:
        B = np.copy(image[:,:,0])
        R = np.copy(image[:,:,2])
        original = np.copy(image)
        original[:,:,0] = R
        original[:,:,2] = B
        plt.imshow(original)
        plt.show()
        if i > 10:
             debug = False
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)

    # Detect Face(s)
    with torch.no_grad():

        # Preprocess image
        input = torch.tensor(image, dtype=torch.float32)

        # Send to GPU
        input = input.to(device)
        input = input - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)

        # Inference
        output = detection_model(input)

        # Post-process output
        for i in range(len(output) // 2):
                output[i * 2] = torch.nn.functional.softmax(output[i * 2], dim=1)
        output = [oelem.data.cpu().numpy() for oelem in output]
        bboxlists = utilities.get_predictions(output, 1)
        bboxlist = utilities.filter_bboxes(bboxlists[0], 0.5)

        # Are there faces?
        if len(bboxlist) > 0:

            # Save Bounding Box
            bbox = bboxlist[0]  # Select the first bounding box
            left= bbox[0]       # Left border (in pixels)
            top = bbox[1]       # Top border (in pixels)
            right = bbox[2]     # Right border (in pixels)
            bottom = bbox[3]    # Bottom border (in pixels)
            file_path = image_path[:-4] + '.txt'    
            file = open(file_path, "w")
            file.write(f"{left:.2f},{top:.2f},{right:.2f},{bottom:.2f}\n")
            file.close()

            # Debug
            if debug:
                plt.imshow(original)
                cmap = colormaps['tab20']
                colors = cmap(np.linspace(0, 1, len(bboxlist)))
                for i, bbox in enumerate(bboxlist):
                    left= bbox[0]
                    top = bbox[1]
                    right = bbox[2]
                    bottom = bbox[3]
                    ax = plt.gca()
                    rectangle = patches.Rectangle((left, top), right-left, bottom-top, linewidth=2, edgecolor=colors[i], facecolor='none')
                    ax.add_patch(rectangle)
                plt.show()
#FIN
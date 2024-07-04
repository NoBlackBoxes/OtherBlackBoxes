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
import detection.model as d_model
import detection.utilities as d_utilities
import alignment.model as a_model
import alignment.utilities as a_utilities
import training.model as t_model
import training.utilities as t_utilities
from importlib import reload
import matplotlib.pyplot as plt

# Debug
debug = True

# Specify paths
box_path = base_path
video_path = base_path + '/_tmp/dataset/C/raw/adam_hof.mp4'
video_name = "adam_hof"
output_folder = base_path + '/_tmp/dataset/C/swap'
detection_model_path = box_path + '/_tmp/models/detection.pth'
alignment_model_path = box_path + '/_tmp/models/alignment.pth'
training_model_path = box_path + '/_tmp/models/training.pth'

# Load models
reload(d_model)
reload(a_model)
reload(t_model)
detection_model = d_model.model()
detection_model.load_state_dict(torch.load(detection_model_path))
alignment_model = a_model.model()
alignment_model.load_state_dict(torch.load(alignment_model_path))
training_model = t_model.model()
training_model.load_state_dict(torch.load(training_model_path))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
detection_model.to(device)
alignment_model.to(device)
training_model.to(device)

# Open Input Video
video = cv2.VideoCapture(video_path)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames = 600

# Open Output Video
output_video_path = base_path + f"/_tmp/dataset/C/swap/{video_name}_swap.avi"
output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('F','M','P','4'), 30, (width, height))

# Extract and swap faces
for i, f in enumerate(range(0,num_frames, 1)):
    print(f"{f} of {num_frames}")
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, image = video.read()
    original = np.copy(image)
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)

    # ---------------
    # Detect the face
    # ---------------
    with torch.no_grad():

        # Preprocess image
        input = torch.tensor(image, dtype=torch.float32)

        # Send to GPU
        input = input.to(device)
        input = input - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)

        # Inference
        output = detection_model(input)

        # Post-process output
        for j in range(len(output) // 2):
                output[j * 2] = torch.nn.functional.softmax(output[j * 2], dim=1)
        output = [oelem.data.cpu().numpy() for oelem in output]
        bboxlists = d_utilities.get_predictions(output, 1)
        bboxlist = d_utilities.filter_bboxes(bboxlists[0], 0.5)

        # Are there faces?
        if len(bboxlist) > 0:

            # Extract Bounding Box
            bbox = bboxlist[0]  # Select the first bounding box
            left= bbox[0]       # Left border (in pixels)
            top = bbox[1]       # Top border (in pixels)
            right = bbox[2]     # Right border (in pixels)
            bottom = bbox[3]    # Bottom border (in pixels)

            # Enforce boundary conditions
            if (left < 0):
                 left = 0
            if (right > (width-1)):
                 right = (width-1)
            if (top < 0):
                 top = 0
            if (bottom > (height-1)):
                 bottom = (height-1)
        else:
             print("No face found!")
             continue

    # --------------
    # Align the face
    # --------------
    left = round(bbox[0])
    top = round(bbox[1])
    right = round(bbox[2])
    bottom = round(bbox[3])
    box_width = right-left
    box_height = bottom-top
    if box_width > box_height:    
        extra_height = int((box_width-box_height)/2)
        new_left = left
        new_right = right
        new_top = top - extra_height
        new_bottom = bottom + extra_height
    else:
        extra_width = int((box_height-box_width)/2)
        new_left = left - extra_width
        new_right = right + extra_width
        new_top = top
        new_bottom = bottom

    # Crop face, resize
    cropped = original[new_top:new_bottom, new_left:new_right]
    x_scale = cropped.shape[1]/256
    y_scale = cropped.shape[0]/256
    resized = cv2.resize(cropped, (256, 256))
    image = resized.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)

    # Detect Face Keypoints (68)
    with torch.no_grad():

        # Preprocess image
        input = torch.tensor(image, dtype=torch.float32) / 255.0

        # Send to GPU
        input = input.to(device)

        # Inference
        heatmaps, stem_feats, hg_feats = alignment_model(input)
        
        # Post-process keypoints
        landmarks, landmark_scores = a_utilities.decode(heatmaps)    
        landmark = landmarks[0]
        hh, hw = heatmaps.size(2), heatmaps.size(3)
        landmark[:, 0] = landmark[:, 0] * 4
        landmark[:, 1] = landmark[:, 1] * 4

        # Create mask from landmarks
        kps = []
        for k in range(68):
             kx = round(landmark[k,0])
             ky = round(landmark[k,1])
             kps.append((kx,ky))
        kps = np.array(kps)
        hull = cv2.convexHull(kps, False)
        mask = np.zeros((256,256), dtype=np.uint8)
        mask = cv2.drawContours(mask, [hull], 0, (1,1,1), cv2.FILLED, 8)

    # --------------
    # Swap the faces
    # --------------
    image = cv2.resize(resized, (64,64))
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)
    input = torch.tensor(image, dtype=torch.float32) / 255.0
    input = input.to(device)
    outputs = training_model(input, 'B')
    outputs = outputs.cpu().detach().numpy()
    output = outputs[0]
    output = np.transpose(output, (1,2,0))
    output = np.uint8(output * 255.0)
    bbox_width = new_right-new_left
    bbox_height = new_bottom-new_top
    resized = cv2.resize(output, (bbox_width,bbox_height))
    mask = cv2.resize(mask, (bbox_width,bbox_height))
    for r in range(bbox_height):
         for c in range (bbox_width):
              if mask[r,c] == 1:
                   original[r+new_top,c+new_left] = resized[r,c]
    output_path = output_folder + f"/{video_name}_{i:04d}.jpg"
    ret = cv2.imwrite(output_path, original)
    ret = output_video.write(original)

# Release the caputre
video.release()
output_video.release()

#FIN
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
import alignment.model as model
import alignment.utilities as utilities
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches

# Debug
debug = True

# Specify paths
box_path = base_path
model_path = box_path + '/_tmp/models/alignment.pth'
input_folder = base_path + '/_tmp/dataset/D'

# Load model
alignment_model = model.model()
alignment_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
alignment_model.to(device)

# Find all input files (bounding box text files)
face_paths = glob.glob(input_folder + "/*.txt")
num_faces = len(face_paths)

# Detect face keypoints and save cropped/aligned/resize images
for i, face_path in enumerate(face_paths):
    print(f"{i} of {num_faces}")
    image_path = face_path[:-4] + '.jpg'

    # Load bounding box coordinates
    bbox = np.genfromtxt(face_path, delimiter=",")

    # Crop and resize image
    image = cv2.imread(image_path)
    width = image.shape[1]
    height = image.shape[0]
    original = np.copy(image)
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

    # Expand boundary by 5% of face scale
    face_scale = ((new_right-new_left) + (new_bottom-new_top)) / 2
    border = round(face_scale * 0.05)
    new_left = new_left - border
    new_right = new_right + border
    new_top = new_top - border
    new_bottom = new_bottom + border

    # Enforce boundary conditions
    if (new_left < 0):
            new_left = 0
    if (new_right > (width-1)):
            new_right = (width-1)
    if (new_top < 0):
            new_top = 0
    if (new_bottom > (height-1)):
            new_bottom = (height-1)

    # Crop face, resize
    cropped = image[new_top:new_bottom, new_left:new_right]
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
        landmarks, landmark_scores = utilities.decode(heatmaps)    
        landmark = landmarks[0]
        hh, hw = heatmaps.size(2), heatmaps.size(3)
        landmark[:, 0] = landmark[:, 0] * 4
        landmark[:, 1] = landmark[:, 1] * 4

        ## Compute angle
        #left_eye = np.mean(landmark[36:42], axis=0)
        #right_eye = np.mean(landmark[42:48], axis=0)
        #dX = right_eye[0] - left_eye[0]
        #dY = right_eye[1] - left_eye[1]
        #tangent = -dY/dX
        #angle = np.arctan(tangent) * (360.0 / (2.0 * 3.1415926))

        ## Find average color
        #background = np.mean(np.mean(original, axis=0), axis=0)

        ## Rotate image
        #cx = 128
        #cy = 128
        #M = cv2.getRotationMatrix2D((cx,cy), -1*angle, 1.0)
        #aligned = cv2.warpAffine(resized, M, (256,256), borderValue=background)

        # Save aligned face
        aligned = np.copy(resized)
        aligned_path = face_path[:-4] + '_aligned.jpg'
        ret = cv2.imwrite(aligned_path, aligned)

        # Display
        if debug:
            x_landmarks = (x_scale * landmark[:, 0]) + new_left
            y_landmarks = (y_scale * landmark[:, 1]) + new_top
            plt.subplot(1,2,1)
            plt.imshow(original[:,:,::-1])
            plt.plot(x_landmarks[:], y_landmarks[:], 'y.')
            plt.plot(x_landmarks[36:42], y_landmarks[36:42], 'g.')
            plt.plot(x_landmarks[42:48], y_landmarks[42:48], 'r.')
            plt.subplot(1,2,2)
            plt.imshow(aligned[:,:,::-1])
            plt.show()
            if i >= 2:
             debug = False

#FIN
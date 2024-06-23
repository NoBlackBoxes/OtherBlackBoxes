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
import alignment.model as model
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Debug
debug = True

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/OtherBlackBoxes'
box_path = repo_path + '/ai/deepfakes'
model_path = box_path + '/_tmp/models/detection.pth'

# Load model
model = model.custom()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)

# Load test image
image_path = repo_path + '/ai/transformers/_data/zoom_lesson.jpg'
image = Image.open(image_path)
if debug:
    plt.imshow(image)
    plt.show()

# Specify transforms for inputs
preprocess = transforms.Compose([
    transforms.ToTensor(),
])

# Preprocess image
input = preprocess(image)

# Send to GPU
input = input.to(device)

# Inference
output = custom_model(input)


bboxlist = output[0].cpu()

if len(bboxlist) > 0:
    keep = nms(bboxlist, 0.3)
    bboxlist = bboxlist[keep, :]
    bboxlist = [x for x in bboxlist if x[-1] > self.fiter_threshold]


#FIN
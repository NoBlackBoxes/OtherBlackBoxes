import os
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from PIL import Image

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/OtherBlackBoxes'
box_path = repo_path + '/ai/transformers/vision/captions'
#image_path = box_path + '/_data/bootcampers.jpg'
image_path = box_path + '/_data/books.jpg'

# Load model and data (pre)processor
checkpoint = "microsoft/git-base"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)

# Set compute device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load image
image = Image.open(image_path)

# Process image for network input
inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

# Send to input and model to device
model = model.to(device)
pixel_values = pixel_values.to(device)

# Run model
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)

#FIN
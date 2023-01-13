from PIL import Image
import requests
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from transformers import AutoFeatureExtractor, BeitForSemanticSegmentation
import torch

# Specify paths
repo = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
image_path = repo + '/boxes/ai/transformers/_data/zoom_lesson.jpg'
image = Image.open(image_path)

# Display test image
plt.figure()
plt.imshow(image)
plt.show()

# Download feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

# Extract features (resizes and normalizes)
encoding = feature_extractor(image, return_tensors="pt")
print(encoding['pixel_values'].shape)

# Download model (900 MB)
model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

# Run model
outputs = model(**encoding)
logits = outputs.logits
output = torch.sigmoid(logits).detach().numpy()[0]
output = np.transpose(output, (1,2,0))

a = cv2.resize(output, (1280,960))

# Monitor
plt.imshow(a[:,:, 143])
plt.show()

#FIN
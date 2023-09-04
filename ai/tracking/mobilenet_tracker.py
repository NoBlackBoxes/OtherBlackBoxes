from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
video_path = repo_path + '/boxes/ai/tracking/_data/water_bottle.mp4'
model_path = '/home/kampff/Dropbox/Voight-Kampff/Technology/mobilenet2-7.onnx'
labels_path = repo_path + '/boxes/ai/tracking/_data/imagenet_labels.txt'

# Load ImageNet class labels
class_labels = np.array(open(labels_path).read().splitlines())

# Load model
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

# Report layer names
for name, layer in model.named_children():
    layer.__name__ = name
    print(name)

# Get classifier weights
classifier_weights = model.classifier[1].weight.cpu().detach().numpy()

# Register hook for activations
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.features[18].register_forward_hook(get_activation('cam'))

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    model.to('cuda')

# Open video
video = cv2.VideoCapture(video_path)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create named window for diaply
cv2.namedWindow('preview')
cv2.moveWindow("preview", 20, 20)

# Select class
class_id = 838-1

# Track
for f in range(0, num_frames):
    ret, im = video.read()
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224,224))
    resized_float = resized.astype(np.float32)
    data = np.copy(resized_float.transpose([2, 0, 1])) # transpose to Channel * Height * Width
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(data.shape[0]):
        data[channel, :, :] = (data[channel, :, :] / 255 - mean[channel]) / std[channel]
    data = np.expand_dims(data, 0)
    input = torch.from_numpy(data)

    # move the input and model to GPU for speed if available
    input_batch = input.to('cuda')

    # Run model
    with torch.no_grad():
        output = model(input_batch)

    # Compute class activation maps
    map = np.zeros((7,7))
    activations = np.squeeze(activation['cam'].cpu().detach().numpy())
    for index, weight in enumerate(classifier_weights[class_id, :]):
        map[:,:] = map[:,:] + (weight * activations[index,:,:])

    # Display results
    overlay = cv2.resize(map, (224,224))
    overlay = np.clip(overlay, 0, 50)
    overlay = (overlay * 5).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    cv2.imshow('preview', np.hstack((resized, overlay)))
    ret = cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()

# FIN
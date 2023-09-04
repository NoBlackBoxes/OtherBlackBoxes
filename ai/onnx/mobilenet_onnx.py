from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnxruntime

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
image_path = repo_path + '/boxes/ai/onnx/_data/sunflowers_pumpkins.jpg'
model_path = '/home/kampff/Dropbox/Voight-Kampff/Technology/Models/mobilenet_v2_1.0_224/model.onnx'
labels_path = repo_path + '/boxes/ai/onnx/_data/imagenet_labels.txt'

# Helper functions
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Load test image
image = Image.open(image_path)

# Load ImageNet class labels
class_labels = np.array(open(labels_path).read().splitlines())

# Preprocess (rescale, cast, normalize, and arrange as tensor)
rescaled = image.resize((224,224), Image.ANTIALIAS)
rescaled_float = np.array(rescaled).astype(np.float32)
data = np.copy(rescaled_float.transpose([2, 0, 1])) # transpose to Channel * Height * Width
mean = np.array([0.079, 0.05, 0]) + 0.406
std = np.array([0.005, 0, 0.001]) + 0.224
for channel in range(data.shape[0]):
    data[channel, :, :] = (data[channel, :, :] / 255 - mean[channel]) / std[channel]
input = np.expand_dims(data, 0)

# Display test image
plt.figure()
plt.imshow(rescaled_float/255.0)
plt.show()

# Inspect model
model = onnx.load(model_path)
for node in model.graph.node:
  print(node.name)

# Load model (into ONNX session)
session = onnxruntime.InferenceSession(model_path)

# Run model
output = session.run([], {'pixel_values':input})[0]

# Convert to class probabilities
output = output.flatten()
output = softmax(output)

# Determine top 5 classes
top_5 = np.argsort(output)[::-1][:5]

# Plot output
plt.figure()
plt.plot(output)
plt.show()
predicted_label = np.argmax(output)
for i in range(5):
    print("Predicted labels: {0}".format(class_labels[top_5[i]]))

# FIN
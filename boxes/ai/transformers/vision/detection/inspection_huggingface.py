from socket import AF_BRIDGE
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch

# Specify paths
repo = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
image_path = repo + '/boxes/ai/transformers/_data/zoom_lesson.jpg'
image = Image.open(image_path)

## Display test image
#plt.figure()
#plt.imshow(image)
#plt.show()

# Download feature extractor
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

# Extract features (resizes and normalizes)
encoding = feature_extractor(image, return_tensors="pt")
print(encoding['pixel_values'].shape)

# Download model (167 MB)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Run model
outputs = model(**encoding)

# Only keep predictions of queries with >0.9 confidence (excluding no-object class)
probabilities = outputs.logits.softmax(-1)[0, :, :-1]
valid = probabilities.max(-1).values > 0.95
valid_probs = probabilities[valid]

# Rescale bounding boxes
target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
valid_bboxes = postprocessed_outputs[0]['boxes'][valid].detach().numpy()

# Plot results
plt.figure(figsize=(16,10))
plt.imshow(image)
ax = plt.gca()
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
colors = COLORS * 100
for p, (xmin, ymin, xmax, ymax), c in zip(valid_probs, valid_bboxes.tolist(), colors):
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
    cl = p.argmax()
    text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
    ax.text(xmin, ymin, text, fontsize=15,
            bbox=dict(facecolor='yellow', alpha=0.5))
plt.axis('off')
plt.show()

# use lists to store the outputs via up-values
conv_features = []

hooks = [model.model.backbone.conv_encoder.register_forward_hook(lambda self, input, output: conv_features.append(output)),]

# propagate through the model
outputs = model(**encoding, output_attentions=True)

for hook in hooks:
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0]
# get cross-attention weights of last decoder layer - which is of shape (batch_size, num_heads, num_queries, width*height)
dec_attn_weights = outputs.cross_attentions[-1]
# average them over the 8 heads and detach from graph
dec_attn_weights = torch.mean(dec_attn_weights, dim=1).detach()


# get the feature map shape
h, w = conv_features[-1][0].shape[-2:]

fig, axs = plt.subplots(ncols=len(valid_bboxes), nrows=2, figsize=(22, 7))
colors = COLORS * 100
for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(valid.nonzero(), axs.T, valid_bboxes):
    ax = ax_i[0]
    ax.imshow(dec_attn_weights[0, idx].view(h, w))
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}')
    ax = ax_i[1]
    ax.imshow(image)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color='blue', linewidth=3))
    ax.axis('off')
    ax.set_title(model.config.id2label[probabilities[idx].argmax().item()])
fig.tight_layout()
plt.show()     

#FIN
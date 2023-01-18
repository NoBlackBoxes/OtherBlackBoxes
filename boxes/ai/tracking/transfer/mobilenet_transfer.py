from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch
from torchvision import transforms

# Locals libs
import model
import dataset

# Reimport
import importlib
importlib.reload(model)
importlib.reload(dataset)

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
coco_folder = '/home/kampff/Dropbox/Voight-Kampff/Technology/Datasets/coco'
dataset_name = 'val2017'
annotations_path = coco_folder + '/annotations/person_keypoints_val2017.json'
images_folder = coco_folder + '/val2017'

# Initialize the COCO API
coco=COCO(annotations_path)

# Select all people: category and image IDs
cat_ids = coco.getCatIds(catNms=['person'])
img_ids = coco.getImgIds(catIds=cat_ids )

# Select annotations of images with only one person with a visible nose
image_paths = []
targets = []
for img in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img, catIds=cat_ids, iscrowd=None)
    annotations = coco.loadAnns(ann_ids)

    # Individuals
    if len(annotations) > 1:
        continue

    # No crowds
    if annotations[0]['iscrowd']:
        continue

    # Only visible noses
    if (annotations[0]['keypoints'][2] == 0):
        continue
    
    # Isolate image path
    img = coco.loadImgs(annotations[0]['image_id'])[0]

    # Normalize nose centroid
    width = img['width']
    height = img['height']
    x = np.float32(annotations[0]['keypoints'][0]) / width
    y = np.float32(annotations[0]['keypoints'][1]) / height
    target = np.array([x, y], dtype=np.float32)

    # Store dataset
    image_paths.append(images_folder + '/' + img['file_name'])
    targets.append(target)

# Specify transforms for inputs
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset and data loader
train_dataset = dataset.custom(image_paths=image_paths, targets=targets, transform=preprocess)
test_dataset = dataset.custom(image_paths=image_paths, targets=targets, transform=preprocess)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Instantiate model
custom_model = model.custom()

# Set loss function
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.00001)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = custom_model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# TRAIN
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_fn, optimizer)
print("Done!")


## Display image and label.
train_features, train_targets = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Targets batch shape: {train_targets.size()}")

# Let's run it
output = custom_model(train_features)

# Examine
for i in range(32):
    feature = train_features[i]
    target = train_targets[i]
    feature = (feature + 1.0) / 2.0
    image = np.transpose(feature, (1,2,0))
    plt.imshow(image)
    plt.plot(target[0] * 224, target[1] * 224, 'go', markersize=15, fillstyle='full')
    plt.show()

# FIN
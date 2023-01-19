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
box_path = repo_path + '/boxes/ai/tracking/transfer'
output_path = box_path + '/_tmp'
coco_folder = '/home/kampff/Dropbox/Voight-Kampff/Technology/Datasets/coco'
dataset_name = 'train2017'
annotations_path = coco_folder + '/annotations/person_keypoints_' + dataset_name + '.json'
images_folder = coco_folder + '/' + dataset_name

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

    # Not too small
    if (annotations[0]['area'] < 10000):
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

# Train/Test Split
num_samples = len(targets)
num_train = int(0.9 * num_samples)
num_test = num_samples - num_train
train_image_paths = image_paths[:num_train]
train_targets = targets[:num_train]
test_image_paths = image_paths[num_train:]
test_targets = targets[num_train:]

# Specify transforms for inputs
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets
train_dataset = dataset.custom(image_paths=train_image_paths, targets=train_targets, transform=preprocess)
test_dataset = dataset.custom(image_paths=test_image_paths, targets=test_targets, transform=preprocess)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)

# Instantiate model
custom_model = model.custom()

# Set loss function
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.0001)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)

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
            pixel_loss = np.sqrt(loss) * 224.0
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], pixel_loss: {pixel_loss:>5f}")

# Define testing
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

# TRAIN
epochs = 25
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_fn, optimizer)
    test(test_dataloader, custom_model, loss_fn)
print("Done!")

# Display image and label.
train_features, train_targets = next(iter(test_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Targets batch shape: {train_targets.size()}")

# Let's run it
train_features_gpu = train_features.to(device)
outputs = custom_model(train_features_gpu)
outputs = outputs.cpu().detach().numpy()

# Examine predictions
for i in range(9):
    plt.subplot(3,3,i+1)
    feature = train_features[i]
    target = train_targets[i]
    output = outputs[i]
    feature = (feature + 1.0) / 2.0
    image = np.transpose(feature, (1,2,0))
    plt.imshow(image)
    plt.plot(output[0] * 224, output[1] * 224, 'yo', markersize=15, fillstyle='full')
    plt.plot(target[0] * 224, target[1] * 224, 'g+', markersize=15,)
plt.show()

# Save model
torch.save(custom_model.state_dict(), output_path + '/custom.pt')


# FIN
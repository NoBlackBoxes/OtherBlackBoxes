import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
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
annotations_path = coco_folder + '/annotations/instances_' + dataset_name + '.json'
images_folder = coco_folder + '/' + dataset_name

# Initialize the COCO API
coco=COCO(annotations_path)

# Select all planes: category and image IDs
cat_ids = coco.getCatIds(catNms=['stop sign'])
img_ids = coco.getImgIds(catIds=cat_ids)

# Select annotations of images with only one person with a visible nose
image_paths = []
targets = []
for i in img_ids:
    ann_ids = coco.getAnnIds(imgIds=i, catIds=cat_ids, iscrowd=None)
    annotations = coco.loadAnns(ann_ids)

    ## Individuals
    #if len(annotations) > 1:
    #    continue

    # Extract bbox: convert cx,cy,w,h to lr/ul
    bbox = annotations[0]['bbox']
    lr_x = bbox[0] + (bbox[2]/2)
    lr_y = bbox[1] + (bbox[3]/2)
    ul_x = bbox[0] - (bbox[2]/2)
    ul_y = bbox[1] - (bbox[3]/2)

    # Isolate image path
    img = coco.loadImgs(annotations[0]['image_id'])[0]

    # Normalize
    width = img['width']
    height = img['height']
    lr_x = np.float32(lr_x) / width
    lr_y = np.float32(lr_y) / height
    ul_x = np.float32(ul_x) / width
    ul_y = np.float32(ul_y) / height
    target = np.array([lr_x, lr_y, ul_x, ul_y], dtype=np.float32)

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
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

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
    pixel_loss = np.sqrt(test_loss) * 224.0
    print(f"Test Error: \n Avg loss: {test_loss:>8f}, pixel_loss: {pixel_loss:>5f}\n")

# TRAIN
epochs = 25
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_fn, optimizer)
    test(test_dataloader, custom_model, loss_fn)
print("Done!")

#

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
    feature = train_features[i]
    target = train_targets[i]
    output = outputs[i]

    predicted_lr = output[0:2]
    predicted_ul = output[2:4]
    prediced_center = 224 * (predicted_lr + predicted_ul)/2
    predicted_width = (output[0] - output[2]) * 224
    predicted_height = (output[1] - output[3]) * 224

    target_lr = target[0:2]
    target_ul = target[2:4]
    target_center = 224 * (target_lr + target_ul)/2
    target_width = (target[0] - target[2]) * 224
    target_height = (target[1] - target[3]) * 224

    feature = (feature + 2.0) / 3.0
    image = np.transpose(feature, (1,2,0))

    ax = plt.subplot(3,3,i+1)
    plt.imshow(image)
    
    predicted_rect = patch.Rectangle((prediced_center[0], prediced_center[1]), predicted_width, predicted_height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(predicted_rect)

    target_rect = patch.Rectangle((target_center[0], target_center[1]), target_width, target_height, linewidth=2, edgecolor='y', facecolor='none')
    ax.add_patch(target_rect)
plt.show()


#

# Save model
torch.save(custom_model.state_dict(), output_path + '/custom.pt')


# FIN
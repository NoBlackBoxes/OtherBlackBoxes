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
from torchsummary import summary
import numpy as np
import glob
import cv2
import training.model as model
import training.dataset as dataset
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches
from importlib import reload

# Debug
debug = True

# Specify paths
box_path = base_path
model_path = box_path + '/_tmp/models/training.pth'
interim_folder = box_path + '/_tmp/models/interim'
dataset_folder_A = base_path + '/_tmp/dataset/A'
dataset_folder_B = base_path + '/_tmp/dataset/B'

# Prepare datasets
reload(dataset)
#train_data_A, test_data_A = dataset.prepare(dataset_folder_A, 0.8)
train_data_B, test_data_B = dataset.prepare(dataset_folder_B, 0.8)

# Create datasets
#train_dataset_A = dataset.dataset(image_paths=train_data_A, augment=False)
#test_dataset_A = dataset.dataset(image_paths=test_data_A, augment=False)
train_dataset_B = dataset.dataset(image_paths=train_data_B, augment=False)
test_dataset_B = dataset.dataset(image_paths=test_data_B, augment=False)

# Create data loaders
#train_dataloader_A = torch.utils.data.DataLoader(train_dataset_A, batch_size=32, shuffle=True)
#test_dataloader_A = torch.utils.data.DataLoader(test_dataset_A, batch_size=32, shuffle=True)
train_dataloader_B = torch.utils.data.DataLoader(train_dataset_B, batch_size=32, shuffle=True)
test_dataloader_B = torch.utils.data.DataLoader(test_dataset_B, batch_size=32, shuffle=True)

# Inspect dataset?
if debug:
    train_features, train_targets = next(iter(train_dataloader_B))
    for i in range(9):
        plt.subplot(3,3,i+1)
        feature = train_features[i]
        image = np.transpose(feature, (1,2,0))
        image = np.uint8(image * 255.0)
        plt.imshow(image)
    plt.show()

# Load model
reload(model)
training_model = model.model()

# Set loss function
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(training_model.parameters(), lr=0.00005, betas=(0.5,0.999))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
training_model.to(device)
summary(training_model, (3, 64, 64))
#summary(training_model, (512, 8, 8))

image_path = "/home/kampff/NoBlackBoxes/OtherBlackBoxes/ai/deepfakes/_tmp/dataset/B/beast_clips_1_aligned.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (64,64))
image = image.transpose(2, 0, 1)
image = np.expand_dims(image, 0)
input = torch.tensor(image, dtype=torch.float32) / 255.0
input = input.to(device)
output = training_model(input)

# Define training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        #print(" - range: {0:.3f} to {1:.3f}".format(pred[0].min(), pred[0].max()))

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
    print(f"Test Error: \n Avg loss: {test_loss:>8f}, pixel_loss: {pixel_loss:>5f}")
    pred = pred.cpu().detach().numpy()
    print(f" - mean prediction: {np.mean(np.mean(np.mean((np.abs(pred)))))}\n")

# TRAIN
epochs = 1000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader_B, training_model, loss_fn, optimizer)
    test(test_dataloader_B, training_model, loss_fn)
    if (t % 10) == 0:
        # Save interim model
        torch.save(training_model.state_dict(), interim_folder + f"/training_{t}.pth")
print("Done!")

# Save final model
torch.save(training_model.state_dict(), model_path)

#FIN
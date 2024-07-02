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
debug = False

# Specify paths
box_path = base_path
model_path = box_path + '/_tmp/models/training.pth'
interim_folder = box_path + '/_tmp/models/interim'
dataset_folder_A = base_path + '/_tmp/dataset/C'
dataset_folder_B = base_path + '/_tmp/dataset/D'

# Prepare datasets
reload(dataset)
train_data_A, test_data_A = dataset.prepare(dataset_folder_A, 0.8)
train_data_B, test_data_B = dataset.prepare(dataset_folder_B, 0.8)

# Create datasets
train_dataset_A = dataset.dataset(image_paths=train_data_A, augment=True)
test_dataset_A = dataset.dataset(image_paths=test_data_A, augment=True)
train_dataset_B = dataset.dataset(image_paths=train_data_B, augment=True)
test_dataset_B = dataset.dataset(image_paths=test_data_B, augment=True)

# Create data loaders
train_dataloader_A = torch.utils.data.DataLoader(train_dataset_A, batch_size=16, shuffle=True)
test_dataloader_A = torch.utils.data.DataLoader(test_dataset_A, batch_size=16, shuffle=True)
train_dataloader_B = torch.utils.data.DataLoader(train_dataset_B, batch_size=16, shuffle=True)
test_dataloader_B = torch.utils.data.DataLoader(test_dataset_B, batch_size=16, shuffle=True)

# Inspect dataset?
if debug:
    train_features, train_targets = next(iter(train_dataloader_A))
    for i in range(9):
        plt.subplot(3,3,i+1)
        feature = train_features[i]
        image = np.transpose(feature, (1,2,0))
        image = np.uint8(image * 255.0)
        plt.imshow(image)
    plt.show()
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
loss_fn_A = torch.nn.L1Loss()
loss_fn_B = torch.nn.L1Loss()
optimizer_A = torch.optim.Adam([{'params': training_model.encoder.parameters()}, {'params': training_model.decoder_A.parameters()}], lr=5e-5, betas=(0.5, 0.999))
optimizer_B = torch.optim.Adam([{'params': training_model.encoder.parameters()}, {'params': training_model.decoder_B.parameters()}], lr=5e-5, betas=(0.5, 0.999))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
training_model.to(device)
summary(training_model, (3, 64, 64))
#summary(training_model, (512, 8, 8))

# Define training
def train(dataloader_A, dataloader_B, batch_size, model, loss_fn_A, loss_fn_B, optimizer_A, optimizer_B):
    size_A = len(dataloader_A.dataset)
    size_B = len(dataloader_B.dataset)
    model.train()
    min_size = min(size_A, size_B)
    num_batches = min_size // batch_size
    for i in range(num_batches):
        features_A, targets_A = next(iter(train_dataloader_A))
        features_B, targets_B = next(iter(train_dataloader_B))
        features_A = features_A.to(device)
        targets_A = targets_A.to(device)
        features_B = features_B.to(device)
        targets_B = targets_B.to(device)

        # Compute prediction error A
        pred_A = model(features_A, select='A')
        loss_A = loss_fn_A(pred_A, targets_A)

        # Compute prediction error B
        pred_B = model(features_B, select='B')
        loss_B = loss_fn_B(pred_B, targets_B)

        # Zero-gradients
        optimizer_A.zero_grad()
        optimizer_B.zero_grad()
        
        # Backpropagation
        loss_A.backward()
        loss_B.backward()

        # Increment weights
        optimizer_A.step()
        optimizer_B.step()

        # Report progress
        if i % 20 == 0:
            print(f"loss_A: {loss_A:>7f} | loss_B: {loss_B:>7f}")

# Define testing
def test(dataloader_A, dataloader_B, batch_size, model, loss_fn_A, loss_fn_B):
    size_A = len(dataloader_A.dataset)
    size_B = len(dataloader_B.dataset)
    model.eval()
    min_size = min(size_A, size_B)
    num_batches = min_size // batch_size
    test_loss_A = 0.0
    test_loss_B = 0.0
    with torch.no_grad():
        for i in range(num_batches):
            features_A, targets_A = next(iter(train_dataloader_A))
            features_B, targets_B = next(iter(train_dataloader_B))
            features_A = features_A.to(device)
            targets_A = targets_A.to(device)
            features_B = features_B.to(device)
            targets_B = targets_B.to(device)

            # Compute prediction error A
            pred_A = model(features_A, select='A')
            loss_A = loss_fn_A(pred_A, targets_A)
            test_loss_A += loss_A

            # Compute prediction error B
            pred_B = model(features_B, select='B')
            loss_B = loss_fn_B(pred_B, targets_B)
            test_loss_B += loss_B

    # Report average test loss
    test_loss_A /= num_batches
    test_loss_B /= num_batches
    print(f"Test Error: \n Avg loss A: {test_loss_A:>8f}, Avg loss B: {test_loss_B:>8f}")
    pred_A = pred_A.cpu().detach().numpy()
    pred_B = pred_B.cpu().detach().numpy()
    print(f" - mean prediction A: {np.mean(np.mean(np.mean((np.abs(pred_A)))))}")
    print(f" - mean prediction B: {np.mean(np.mean(np.mean((np.abs(pred_B)))))}\n")

# TRAIN
epochs = 1000
batch_size = 16
torch.autograd.set_detect_anomaly(True)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader_A, train_dataloader_B, batch_size, training_model, loss_fn_A, loss_fn_B, optimizer_A, optimizer_B)
    test(test_dataloader_A, test_dataloader_B, batch_size, training_model, loss_fn_A, loss_fn_B)
    if (t % 1000) == 0:
        # Save interim model
        torch.save(training_model.state_dict(), interim_folder + f"/training_{t}.pth")
    if (t % 1) == 0:
        # Save latest model
        torch.save(training_model.state_dict(), model_path)

        # Save current preview
        preview_image_path = "/home/kampff/NoBlackBoxes/OtherBlackBoxes/ai/deepfakes/_tmp/dataset/C/adam_intro_4_aligned.jpg"
        preview_output_path = f"/home/kampff/NoBlackBoxes/OtherBlackBoxes/ai/deepfakes/_tmp/dataset/results/{t:04d}.jpg"
        original = cv2.imread(preview_image_path)
        image = cv2.resize(original, (64,64))
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, 0)
        input = torch.tensor(image, dtype=torch.float32) / 255.0
        input = input.to(device)
        outputs = training_model(input, 'B')
        outputs = outputs.cpu().detach().numpy()
        output = outputs[0]
        output = np.transpose(output, (1,2,0))
        output = np.uint8(output * 255.0)
        resized = cv2.resize(output, (256,256))
        display = np.hstack((original,resized))
        cv2.imwrite(preview_output_path, display)
print("Done!")

# Save final model
torch.save(training_model.state_dict(), model_path)

#FIN
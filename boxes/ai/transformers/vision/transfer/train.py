import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import transforms
from torchsummary import summary

# Locals libs
import model
import dataset

# Reimport
import importlib
importlib.reload(dataset)
importlib.reload(model)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/repos/OtherBlackBoxes'
box_path = repo_path + '/boxes/ai/transformers/vision/transfer'
output_path = box_path + '/_tmp'

# Specify transforms for inputs
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prepare datasets
train_data, test_data = dataset.prepare('train2017', 0.8)

# Create datasets
train_dataset = dataset.custom(image_paths=train_data[0], targets=train_data[1], transform=preprocess, augment=True)
test_dataset = dataset.custom(image_paths=test_data[0], targets=test_data[1], transform=preprocess, augment=True)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)

# Inspect dataset?
inspect = False
if inspect:
    train_features, train_targets = next(iter(train_dataloader))
    for i in range(9):
        plt.subplot(3,3,i+1)
        feature = train_features[i]
        target = train_targets[i]
        feature = (feature + 2.0) / 4.0
        image = np.transpose(feature, (1,2,0))
        plt.imshow(image)
        plt.plot(target[0] * 224, target[1] * 224, 'g+', markersize=15,)
    plt.show()

# Instantiate model
importlib.reload(model)
custom_model = model.custom()

# Set loss function
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.0001)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)
summary(custom_model, (3, 224, 224))

# Define training
def train(dataloader, _model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    _model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        predictions = _model(X)
        train_loss = loss_fn(predictions, y)
        print(" - range: {0:.6f} to {1:.6f}".format(predictions[0].min(), predictions[0].max()))

        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Report
        train_loss, current = train_loss.item(), batch * len(X)
        pixel_loss = np.sqrt(train_loss) * 224.0
        print(f"loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}], pixel_loss: {pixel_loss:>5f}")

# Define testing
def test(dataloader, _model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    _model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            predictions = _model(X)
            test_loss += loss_fn(predictions, y).item()
    test_loss /= num_batches
    pixel_loss = np.sqrt(test_loss) * 224.0
    print(f"Test Error: \n Avg loss: {test_loss:>8f}, pixel_loss: {pixel_loss:>5f}\n")

# TRAIN
epochs = 250
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_function, optimizer)
    test(test_dataloader, custom_model, loss_function)
print("Done!")


# ------------------------------------------------------------------------
# Display image and label.
train_features, train_targets = next(iter(test_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Targets batch shape: {train_targets.size()}")

# Let's run it
train_features_gpu = train_features.to(device)
outputs = custom_model(train_features_gpu)
outputs = outputs.cpu().detach().numpy()

# Examine predictions
plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    feature = train_features[i]
    target = np.squeeze(train_targets[i].numpy())
    feature = (feature + 2.0) / 4.0
    image = np.transpose(feature, (1,2,0))
    target_heatmap = cv2.resize(target, (224,224))
    output = np.squeeze(outputs[i])
    predicted_heatmap = cv2.resize(output, (224,224))
    plt.imshow(image, alpha=0.75)
    plt.imshow(predicted_heatmap, alpha=0.5)
    #plt.imshow(target_heatmap, alpha=0.5)
plt.savefig(output_path + '/result.png')
# ------------------------------------------------------------------------




# Save model
torch.save(custom_model.state_dict(), output_path + '/custom.pt')


# FIN
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary

# Locals libs
import dataset
import model

# Reimport
import importlib
importlib.reload(dataset)
importlib.reload(model)

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/repos/OtherBlackBoxes'
box_path = repo_path + '/boxes/ai/speech/keyword'
output_path = box_path + '/_tmp'
dataset_folder = box_path + '/_tmp/dataset'

# Prepare datasets
train_data, test_data, num_classes = dataset.prepare(dataset_folder, 0.8)

# Create datasets
train_dataset = dataset.custom(wav_paths=train_data[0], targets=train_data[1], num_classes=num_classes, augment=False)
test_dataset = dataset.custom(wav_paths=test_data[0], targets=test_data[1], num_classes=num_classes, augment=False)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

# Inspect dataset?
inspect = False
if inspect:
    train_features, train_targets = next(iter(train_dataloader))
    for i in range(9):
        plt.subplot(3,3,i+1)
        feature = np.squeeze(train_features[i])
        target = train_targets[i]
        plt.imshow(feature, alpha=0.75)
    plt.show()

# Instantiate model
importlib.reload(model)
custom_model = model.custom()

## Reload saved model
#model_path = model_path = box_path + '/_tmp/custom.pt'
#custom_model = model.custom()
#custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Set loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Set optimizer
optimizer = torch.optim.AdamW(custom_model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.1)

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)
summary(custom_model, (1, 32, 37))

# Define accuracy
def measure_accuracy(targets, guesses):

    # Detach
    targets = targets.cpu().detach().numpy()
    guesses = guesses.cpu().detach().numpy()

    # Measure accuracy
    num_guesses = guesses.shape[0]
    correct = 0
    for i in range(num_guesses):
        target = targets[i]
        guess = guesses[i]
        expected = np.argmax(target)
        predicted = np.argmax(guess)
        if expected == predicted:
            correct = correct + 1
    accuracy = 100.0 * correct/num_guesses
    return accuracy

# Define training
def train(_dataloader, _model, _loss_function, _optimizer):
    size = len(_dataloader.dataset)
    _model.train()
    for batch, (X, y) in enumerate(_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = _model(X)
        loss = _loss_function(pred, y)

        # Backpropagation
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X)
            accuracy = measure_accuracy(y, pred)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] accuracy: {accuracy:.2f}%")

# Define testing
def test(_dataloader, _model, _loss_function):
    size = len(_dataloader.dataset)
    num_batches = len(_dataloader)
    _model.eval()
    test_loss = 0.0
    accum_accuracy = 0
    with torch.no_grad():
        for X, y in _dataloader:
            X, y = X.to(device), y.to(device)
            pred = _model(X)
            test_loss += _loss_function(pred, y).item()
            accuracy = measure_accuracy(y, pred)
            accum_accuracy = accum_accuracy + accuracy
    avg_test_loss = test_loss / num_batches
    avg_accuracy =  accum_accuracy / num_batches
    print(f"Test Results: \n Avg loss: {avg_test_loss:>8f}\n Avg Accu: {avg_accuracy:>4f}%\n")

# TRAIN
epochs = 250
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_fn, optimizer)
    test(test_dataloader, custom_model, loss_fn)
print("Done!")

# Save model
torch.save(custom_model.state_dict(), output_path + '/custom.pt')

# Reload saved model
model_path = model_path = box_path + '/_tmp/custom.pt'
custom_model = model.custom()
custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Measure accuracy
train_features, train_targets = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Targets batch shape: {train_targets.size()}")

# Let's run it
train_features_gpu = train_features.to(device)
outputs = custom_model(train_features_gpu)

# Detach
targets = train_targets.cpu().detach().numpy()
guesses = outputs.cpu().detach().numpy()

num_guesses = guesses.shape[0]
correct = 0
for i in range(num_guesses):
    target = train_targets[i]
    guess = guesses[i]
    expected = np.argmax(target)
    predicted = np.argmax(guess)
    if expected == predicted:
        correct = correct + 1
accuracy = 100.0 * correct/num_guesses
print(f"Avg Accu: {accuracy:.3f}%\n")

# FIN
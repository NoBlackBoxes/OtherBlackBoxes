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
train_data, test_data = dataset.prepare(dataset_folder, 0.8)

# Create datasets
train_dataset = dataset.custom(wav_paths=train_data[0], targets=train_data[1], augment=False)
test_dataset = dataset.custom(wav_paths=test_data[0], targets=test_data[1], augment=False)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)

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
loss_fn = torch.nn.BCELoss()

# Set optimizer
optimizer = torch.optim.AdamW(custom_model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.1)

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)
summary(custom_model, (1, 16, 16))

# Define accuracy
def measure_accuracy(targets, guesses):

    # Detach
    targets = targets.cpu().detach().numpy()
    guesses = guesses.cpu().detach().numpy()

    # Measure accuracy
    num_guesses = guesses.shape[0]
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(num_guesses):
        target = targets[i]
        guess = guesses[i]
        expected = target
        predicted = guess[0]
        if (expected == 1.0):
            if (predicted > 0.5):
                true_positives = true_positives + 1
            else:
                false_negatives = false_negatives + 1
        else:
            if (predicted > 0.5):
                false_positives = false_positives + 1
            else:
                true_negatives = true_negatives + 1
    return (true_positives, true_negatives, false_positives, false_negatives)

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
            results = measure_accuracy(y, pred)
            print(results)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define testing
def test(_dataloader, _model, _loss_function):
    size = len(_dataloader.dataset)
    num_batches = len(_dataloader)
    _model.eval()
    test_loss = 0.0
    accum_results = (0,0,0,0)
    with torch.no_grad():
        for X, y in _dataloader:
            X, y = X.to(device), y.to(device)
            pred = _model(X)
            test_loss += _loss_function(pred, y).item()
            results = measure_accuracy(y, pred)
            accum_results = accum_results + results
    avg_test_loss = test_loss / num_batches
    print(results)
    print(f"Test Results: \n Avg loss: {avg_test_loss:>8f}\n")

# TRAIN
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_fn, optimizer)
    test(test_dataloader, custom_model, loss_fn)
print("Done!")

# Save model
torch.save(custom_model.state_dict(), output_path + '/custom.pt')

# Reload saved model
#model_path = model_path = box_path + '/_tmp/custom.pt'
#custom_model = model.custom()
#custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))



# FIN
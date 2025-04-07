import torch
import timm

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Build model
        self.conv1a = torch.nn.Conv2d(in_channels=1, out_channels=48, kernel_size=(7,3), stride=1, padding=0)
        self.relu1a = torch.nn.ReLU(inplace=True)
        self.conv1b = torch.nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(7,3), stride=1, padding=0)
        self.relu1b = torch.nn.ReLU(inplace=True)
        self.maxpool1 = torch.nn.MaxPool2d(2,2)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.conv2a = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3,3), stride=1, padding=0)
        self.relu2a = torch.nn.ReLU(inplace=True)
        self.conv2b = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        self.relu2b = torch.nn.ReLU(inplace=True)
        self.maxpool2 = torch.nn.MaxPool2d(2,2)
        self.dropout2 = torch.nn.Dropout(0.25)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(4032,64)
        self.dropout3 = torch.nn.Dropout(0.25)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 31)
        self.softmax = torch.nn.Softmax(dim=1)
    
    # Forward
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1a(x)
        x = self.relu1a(x)
        x = self.conv1b(x)
        x = self.relu1b(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2a(x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        x = self.relu2b(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout3(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

#FIN
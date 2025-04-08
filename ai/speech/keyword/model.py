import torch
import timm

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Build model
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(20,8), stride=1, padding="same")
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.maxpool1 = torch.nn.MaxPool2d(1,3)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(10,4), stride=1, padding="same")
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.maxpool2 = torch.nn.MaxPool2d(1,1)
        self.flatten = torch.nn.Flatten()
        self.linear3 = torch.nn.Linear(23232,32)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.linear4 = torch.nn.Linear(32, 128)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.linear5 = torch.nn.Linear(128, 37)
    
    # Forward
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        return x

#FIN
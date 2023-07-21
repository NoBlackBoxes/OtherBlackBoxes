import torch
import timm

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Build model
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(2,2)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(6240,128)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Linear(128, 30)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax()
    
    # Forward
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.softmax(x)
        return x

#FIN
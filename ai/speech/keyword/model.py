import torch
import timm

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Build model
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,3), stride=1, padding=0)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.maxpool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7,3), stride=1, padding=0)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.maxpool2 = torch.nn.MaxPool2d(2,2)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(4416,64)
        self.dropout = torch.nn.Dropout(0.25)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 31)
        self.sigmoid = torch.nn.Softmax(dim=0)
    
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
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x

#FIN
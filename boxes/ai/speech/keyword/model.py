import torch
import timm

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Add a new prediction head
        self.first = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.final = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(7200,29)
    
    # Forward
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.first(x)
        x = self.flatten(x)
        x = self.relu(x)
        x = self.linear(x)
        return x

#FIN
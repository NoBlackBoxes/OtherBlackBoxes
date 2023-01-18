import torch

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Load backbone
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

        # Freeze the backbone weights
        for param in backbone.parameters():
            param.requires_grad = False

        # Remove classifier (i.e. extract feature detection layers)
        self.features =  torch.nn.Sequential(*list(backbone.children())[:-2])

        # Add a new prediction head
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(150528, 100)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, 50)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(50, 2)
        self.sigmoid = torch.nn.Sigmoid()


    # Print
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x

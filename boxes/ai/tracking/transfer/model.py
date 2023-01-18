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
        self.features = backbone.features

        # Add a new prediction head
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(62720, 2)

    # Print
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

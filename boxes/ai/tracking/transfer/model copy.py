import torch

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Load backbone
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='MobileNet_V2_Weights.IMAGENET1K_V1')

        # Freeze the backbone weights
        for param in backbone.parameters():
            param.requires_grad = False

        # Remove classifier (i.e. extract feature detection layers)
        self.features =  torch.nn.Sequential(*list(backbone.children())[:-10])

        # Add a new prediction head
        self.conv_upscale = torch.nn.Sequential(
                                torch.nn.ConvTranspose2d(1280, 640, kernel_size=7, stride=1, bias=False),
                                torch.nn.BatchNorm2d(640,eps=0.001,momentum=0.1),
                                torch.nn.ReLU(inplace=True)
        )
        self.conv_last = torch.nn.Sequential(
                                torch.nn.Conv2d(640, 64, kernel_size=1, stride=1, bias=False),
                                torch.nn.BatchNorm2d(64,eps=0.001,momentum=0.1),
                                torch.nn.ReLU(inplace=True)
                            )
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(10816,2)
        self.sigmoid = torch.nn.Sigmoid()

    # Forward
    def forward(self, x):
        #print(x.size())
        x = self.features(x)
        #print(x.size())
        x = self.conv_upscale(x)
        #print(x.size())
        x = self.conv_last(x)
        #print(x.size())
        x = self.flatten(x)
        #print(x.size())
        x = self.linear(x)
        #print(x.size())
        x = self.sigmoid(x)
        #print(x.size())
        return x


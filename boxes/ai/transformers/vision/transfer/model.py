import torch

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Load backbone
        backbone = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

        # Freeze the backbone weights
        for param in backbone.parameters():
            param.requires_grad = True
        
        ## Unfreeze last layers
        #for b in range(11,7):
        #    for param in backbone.blocks[b].parameters():
        #        param.requires_grad = True

        # Remove classifier (i.e. extract feature detection layers)
        self.features =  torch.nn.Sequential(*list(backbone.children())[:-1])

        # Add a new prediction head
        self.pool = torch.nn.AvgPool1d(768)
        self.sigmoid = torch.nn.Sigmoid()

    # Forward
    def forward(self, x):
        n, c, h, w = x.shape
        x = self.features(x)
        x = self.pool(x)
        x = self.sigmoid(x)
        x = x.view(n,1,14,14)

        return x

#FIN
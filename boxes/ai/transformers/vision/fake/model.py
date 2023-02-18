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
        
        # Unfreeze last blocks
        for b in range(5,12):
            for param in backbone.blocks[b].parameters():
                param.requires_grad = False

        # Remove classifier (i.e. extract feature detection layers)
        self.features =  torch.nn.Sequential(*list(backbone.children())[:-2])

        # Add a new prediction head
        self.conv = torch.nn.Conv1d(768, 1, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    # Forward
    def forward(self, x):
        n, c, h, w = x.shape
        x = self.features(x)
        x = x.transpose(2,1)
        x = self.conv(x)
        #x = self.sigmoid(x)
        x = x.view(n,1,14,14)

        return x

#FIN
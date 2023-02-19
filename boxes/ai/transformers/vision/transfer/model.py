import torch
import timm

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Load backbone
        #backbone = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        backbone = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Freeze the backbone weights
        for param in backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last blocks
        for b in range(7,12):
            for param in backbone.blocks[b].parameters():
                param.requires_grad = True

        # Remove classifier (i.e. extract feature detection layers)
        self.features =  torch.nn.Sequential(*list(backbone.children())[:-1])

        # Add a new prediction head
        self.conv1 = torch.nn.Conv2d(768, 64, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.bn2 = torch.nn.BatchNorm2d(1)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.pool = torch.nn.AvgPool1d(768)
        self.linear1 = torch.nn.Linear(196,196*4)
        self.linear2 = torch.nn.Linear(196*4, 196)

    # Forward
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.features(x)
        x = self.pool(x)
        x = x.transpose(2,1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = x.reshape(b, -1, 14, 14).contiguous()

        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu1(x)
        #x = self.conv2(x)
        #x = self.bn2(x)
        #x = self.relu2(x)

        return x

#FIN
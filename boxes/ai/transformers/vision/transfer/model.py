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
            param.requires_grad = True
        
        # Unfreeze last blocks
        for b in range(7,12):
            for param in backbone.blocks[b].parameters():
                param.requires_grad = True

        # Remove classifier (i.e. extract feature detection layers)
        self.features =  torch.nn.Sequential(*list(backbone.children())[:-1])

        # Add a new prediction head
        self.pool = torch.nn.AvgPool1d(768)
        self.linear1 = torch.nn.Linear(768, 1)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 196)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        #self.conv = torch.nn.Conv1d(768, 1, kernel_size=1)
        self.conv = torch.nn.Conv1d(768, 1, kernel_size=2)
        self.sigmoid = torch.nn.Sigmoid()

    # Forward
    def forward(self, x):
        n, c, h, w = x.shape
        x = self.features(x)
        x = self.pool(x)
        #x = x.transpose(2,1)
        #x = self.linear1(x)
        #x = self.relu1(x)
        #x = self.linear2(x)
        #x = self.relu2(x)
        #x = self.linear3(x)

        #x = x.transpose(2,1)
        #x = self.conv(x)
        
        #x = x.transpose(2,1)
        #x = self.linear1(x)

        x = self.sigmoid(x) * 100
        x = x.view(n,1,14,14)

        return x

#FIN
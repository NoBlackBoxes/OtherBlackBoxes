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

        avail_pretrained_models = timm.list_models(pretrained=True)
        len(avail_pretrained_models), avail_pretrained_models[:5]

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
        #self.conv1 = torch.nn.Conv2d(768, 64, kernel_size=1)
        #self.bn1 = torch.nn.BatchNorm2d(64)
        #self.relu1 = torch.nn.ReLU(inplace=True)
        #self.conv2 = torch.nn.Conv2d(64, 1, kernel_size=1)
        #self.bn2 = torch.nn.BatchNorm2d(1)
        #self.relu2 = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
        #self.pool = torch.nn.AvgPool1d(768)
        #self.linear1 = torch.nn.Linear(196,196*4)
        #self.linear2 = torch.nn.Linear(196*4, 196)
        self.relu = torch.nn.ReLU(inplace=True)

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=768,
                                   out_channels=256,
                                   kernel_size=4,
                                   stride=2,
                                   padding=0,
                                   output_padding=1,
                                   bias=False)

        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=256,
                                   out_channels=256,
                                   kernel_size=4,
                                   stride=2,
                                   padding=0,
                                   output_padding=1,
                                   bias=False)

        self.final = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)

    # Forward
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.features(x)
        #x = self.pool(x)
        #x = x.transpose(2,1)
        #x = self.sigmoid(x)
        x = x.reshape(b, -1, 14, 14).contiguous()
        x = self.deconv1(x)
        x = self.relu(x)
        #x = self.deconv2(x)
        #x = self.relu(x)
        x = self.final(x)
        #x = self.relu(x)
        return x

#FIN
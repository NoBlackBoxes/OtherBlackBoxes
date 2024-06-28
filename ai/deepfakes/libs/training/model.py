import torch

# Define encoder
class encoder(torch.nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        self.conv_1 = torch.nn.Conv2d(3, 128, kernel_size=5, stride=2)
        self.relu_1 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_2 = torch.nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.relu_2 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_3 = torch.nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.relu_3 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_4 = torch.nn.Conv2d(512, 1024, kernel_size=5, stride=2)
        self.relu_4 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.flatten = torch.nn.Flatten()
        self.linear_1 = torch.nn.Linear(self.encoder_dim * 4 * 4, self.encoder_dim)
        self.linear_2 = torch.nn.Linear(self.encoder_dim, self.encoder_dim * 4 * 4)
        self.conv_5  = torch.nn.Conv2d(1024, 2048, kernel_size=3)
        self.relu_5 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pixel_shuffle = torch.nn.PixelShuffle(4)

    # Forward    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.conv_4(x)
        x = self.relu_4(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.conv_5(x)
        x = self.relu_5(x)
        x = self.pixel_shuffle(x)
        return x

# Define decoder
class decoder(torch.nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.conv_1 = torch.nn.Conv2d(3, 128, kernel_size=5, stride=2)
        self.relu_1 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_2 = torch.nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.relu_2 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_3 = torch.nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.relu_3 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_4 = torch.nn.Conv2d(512, 1024, kernel_size=5, stride=2)
        self.relu_4 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.faltten = torch.nn.Flatten()
        self.linear_1 = torch.nn.Linear(self.encoder_dim * 4 * 4, self.encoder_dim)
        self.linear_2 = torch.nn.Linear(self.encoder_dim, self.encoder_dim * 4 * 4)
        self.conv_5  = torch.nn.Conv2d(1024, 2048, kernel_size=3)
        self.relu_5 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pixel_shuffle = torch.nn.PixelShuffle(4)

    # Forward    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.conv_4(x)
        x = self.relu_4(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.conv_5(x)
        x = self.relu_5(x)
        x = self.pixel_shuffle(x)
        return x

# Define model
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # Attributes
        self.input_shape = (64, 64, 3)
        self.encoder_dim = 1024

        # Encoder
        self.encoder = encoder()

        # Decoder (A)
        self.decoder_A = decoder()
        self.decoder_B = decoder()

    # Forward    
    def forward(self, x, select='A'):
        x = self.encoder(x)
        if select == 'A':
            x = self.decoder_A(x)
        else:
            x = self.decoder_B(x)
        return x

#FIN
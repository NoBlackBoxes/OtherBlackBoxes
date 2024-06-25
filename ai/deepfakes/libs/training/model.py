import torch

# Define model (which extends the NN module)
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # Attributes
        self.input_shape = (64, 64, 3)
        self.encoder_dim = 1024


    # Forward
    def forward(self, x):
        
        return x
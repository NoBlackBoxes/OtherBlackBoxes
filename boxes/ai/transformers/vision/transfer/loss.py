import torch
import torch.nn as nn


class custom_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        diff = (output - target)
        loss = torch.mean(diff**2)

        return loss

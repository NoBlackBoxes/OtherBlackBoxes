import torch
import torch.nn as nn
import numpy as np

class custom_loss(nn.Module):

    def __init__(self):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.criterion = nn.MSELoss()
        self.xinds = torch.Tensor(np.repeat(np.arange(0,31), 31).reshape(31,31).T.flatten()).to(device)
        self.yinds = torch.Tensor(np.repeat(np.arange(0,31), 31)).to(device)

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            interim = 0.
            for b in range(batch_size):
                px  = 31 * torch.mean(self.xinds * heatmap_pred[b,:])
                py  = 31 * torch.mean(self.yinds * heatmap_pred[b,:])
                tx  = 31 * torch.mean(self.xinds * heatmap_gt[b,:])
                ty  = 31 * torch.mean(self.yinds * heatmap_gt[b,:])
                interim += torch.sqrt((px-tx)**2 + (py-ty)**2) / 2.
            print("Pred: {0:.3f}, {1:.3f} : Targ: {2:.3f}, {3:.3f}".format(px, py, tx, ty))
            interim /= batch_size
            loss += interim

        return loss / num_joints

#FIN
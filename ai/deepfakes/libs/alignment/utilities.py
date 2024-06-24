import numpy as np
import torch
from typing import Union, Optional, Tuple

def decode(heatmaps: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    heatmaps = heatmaps.contiguous()
    scores = heatmaps.max(dim=3)[0].max(dim=2)[0]
    gamma = 1.0
    radius = 0.1

    if (radius ** 2 * heatmaps.shape[2] * heatmaps.shape[3] <
            heatmaps.shape[2] ** 2 + heatmaps.shape[3] ** 2):
        # Find peaks in all heatmaps
        m = heatmaps.view(heatmaps.shape[0] * heatmaps.shape[1], -1).argmax(1)
        all_peaks = torch.cat(
            [(m / heatmaps.shape[3]).trunc().view(-1, 1), (m % heatmaps.shape[3]).view(-1, 1)], dim=1
        ).reshape((heatmaps.shape[0], heatmaps.shape[1], 1, 1, 2)).repeat(
            1, 1, heatmaps.shape[2], heatmaps.shape[3], 1).float()

        # Apply masks created from the peaks
        all_indices = torch.zeros_like(all_peaks) + torch.stack(
            [torch.arange(0.0, all_peaks.shape[2],
                            device=all_peaks.device).unsqueeze(-1).repeat(1, all_peaks.shape[3]),
                torch.arange(0.0, all_peaks.shape[3],
                            device=all_peaks.device).unsqueeze(0).repeat(all_peaks.shape[2], 1)], dim=-1)
        heatmaps = heatmaps * ((all_indices - all_peaks).norm(dim=-1) <= radius *
                                (heatmaps.shape[2] * heatmaps.shape[3]) ** 0.5).float()

    # Prepare the indices for calculating centroids
    x_indices = (torch.zeros((*heatmaps.shape[:2], heatmaps.shape[3]), device=heatmaps.device) +
                    torch.arange(0.5, heatmaps.shape[3], device=heatmaps.device))
    y_indices = (torch.zeros(heatmaps.shape[:3], device=heatmaps.device) +
                    torch.arange(0.5, heatmaps.shape[2], device=heatmaps.device))

    # Finally, find centroids as landmark locations
    heatmaps = heatmaps.clamp_min(0.0)
    if gamma != 1.0:
        heatmaps = heatmaps.pow(gamma)
    m00s = heatmaps.sum(dim=(2, 3)).clamp_min(torch.finfo(heatmaps.dtype).eps)
    xs = heatmaps.sum(dim=2).mul(x_indices).sum(dim=2).div(m00s)
    ys = heatmaps.sum(dim=3).mul(y_indices).sum(dim=2).div(m00s)

    lm_info = torch.stack((xs, ys, scores), dim=-1).cpu().numpy()
    return lm_info[..., :-1], lm_info[..., -1]
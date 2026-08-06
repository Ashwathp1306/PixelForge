import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM Loss (1.0 - SSIM).
    Measures structural similarity between prediction and target.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, max_val: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.max_val = max_val

    def _create_window(self, channels: int, device: torch.device) -> torch.Tensor:
        gauss = torch.exp(-torch.tensor([(x - self.window_size // 2) ** 2 for x in range(self.window_size)]) / (2 * self.sigma ** 2))
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channels, 1, self.window_size, self.window_size).contiguous()
        return window.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.size(1)
        window = self._create_window(channels, pred.device)
        
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channels) - mu1_mu2

        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()

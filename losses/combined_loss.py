import torch
import torch.nn as nn
from typing import Dict, Tuple
from .charbonnier import CharbonnierLoss
from .ssim_loss import SSIMLoss
from .perceptual import PerceptualLoss

class CompositeRestorationLoss(nn.Module):
    """
    Composite Loss Function for Image Restoration:
    L_total = lambda_pixel * L_charbonnier + lambda_ssim * L_ssim + lambda_perceptual * L_perceptual
    """
    def __init__(
        self,
        lambda_pixel: float = 1.0,
        lambda_ssim: float = 0.2,
        lambda_perceptual: float = 0.05,
        use_perceptual: bool = False
    ):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual

        self.charbonnier_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        
        if use_perceptual:
            try:
                self.perceptual_loss = PerceptualLoss()
            except Exception as e:
                print(f"Warning: Failed to load VGG for PerceptualLoss ({e}). Disabling perceptual loss.")
                self.use_perceptual = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_pixel = self.charbonnier_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        
        total_loss = self.lambda_pixel * l_pixel + self.lambda_ssim * l_ssim
        loss_dict = {
            "loss_pixel": l_pixel.item(),
            "loss_ssim": l_ssim.item(),
        }

        if self.use_perceptual and self.lambda_perceptual > 0:
            l_perceptual = self.perceptual_loss(pred, target)
            total_loss += self.lambda_perceptual * l_perceptual
            loss_dict["loss_perceptual"] = l_perceptual.item()

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict

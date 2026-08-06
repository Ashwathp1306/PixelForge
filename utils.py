import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two PyTorch tensors.
    Assumes img1 and img2 are normalized to range [0, max_val] with shape (B, C, H, W) or (C, H, W).
    """
    img1 = torch.clamp(img1, 0.0, max_val)
    img2 = torch.clamp(img2, 0.0, max_val)
    
    mse = F.mse_loss(img1, img2, reduction='mean').item()
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((max_val ** 2) / mse)

def gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    """Generate 2D Gaussian window kernel for SSIM."""
    gauss = torch.exp(-torch.tensor([(x - window_size // 2) ** 2 for x in range(window_size)]) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
    return window.to(device)

def calculate_ssim_tensor(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> float:
    """
    Calculate Differentiable Structural Similarity Index (SSIM) between two PyTorch tensors.
    Accepts (B, C, H, W) or (C, H, W).
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
        
    img1 = torch.clamp(img1, 0.0, max_val)
    img2 = torch.clamp(img2, 0.0, max_val)
    
    channels = img1.size(1)
    window = gaussian_window(window_size, 1.5, channels, img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2
    
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def plot_comparison_grid(
    gt_img: np.ndarray,
    corrupt_img: np.ndarray,
    restored_img: np.ndarray,
    psnr_val: Optional[float] = None,
    ssim_val: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plots a 1x3 triplet comparison grid: Ground Truth | Corrupt (Upsampled for display) | Restored
    Accepts (H, W, C) or (H, W) numpy arrays in range [0, 1].
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Handle single channel / 3 channel display
    cmap = 'gray' if gt_img.ndim == 2 or gt_img.shape[2] == 1 else None
    gt_disp = gt_img.squeeze() if gt_img.ndim == 3 and gt_img.shape[2] == 1 else gt_img
    corrupt_disp = corrupt_img.squeeze() if corrupt_img.ndim == 3 and corrupt_img.shape[2] == 1 else corrupt_img
    restored_disp = restored_img.squeeze() if restored_img.ndim == 3 and restored_img.shape[2] == 1 else restored_img

    # Display images
    axes[0].imshow(np.clip(gt_disp, 0, 1), cmap=cmap)
    axes[0].set_title("Ground Truth (256x256)", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(corrupt_disp, 0, 1), cmap=cmap)
    axes[1].set_title("Corrupt Input (128x128 -> 256x256)", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    title_restored = "Restored Output (256x256)"
    if psnr_val is not None and ssim_val is not None:
        title_restored += f"\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}"
    axes[2].imshow(np.clip(restored_disp, 0, 1), cmap=cmap)
    axes[2].set_title(title_restored, fontsize=12, fontweight='bold', color='darkgreen')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

def plot_intensity_histograms(
    gt_img: np.ndarray,
    corrupt_img: np.ndarray,
    restored_img: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plots pixel intensity distribution comparison histograms:
    - True GT vs Corrupt (raw values, including <0 and >1) vs Restored
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    gt_flat = gt_img.ravel()
    corrupt_flat = corrupt_img.ravel()
    restored_flat = restored_img.ravel()
    
    # Range of histogram binning covering potential negative and >1 noisy LR values
    bins = np.linspace(min(-0.2, corrupt_flat.min()), max(1.2, corrupt_flat.max()), 100)
    
    ax.hist(corrupt_flat, bins=bins, alpha=0.4, color='crimson', label='Corrupt Input y (Noisy LR)', density=True)
    ax.hist(gt_flat, bins=bins, alpha=0.5, color='royalblue', label='Ground Truth x (GT)', density=True)
    ax.hist(restored_flat, bins=bins, alpha=0.6, color='forestgreen', label='Restored Output x̂', density=True)
    
    ax.set_title("Pixel Intensity Distribution Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel("Pixel Intensity Value", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.axvline(0.0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

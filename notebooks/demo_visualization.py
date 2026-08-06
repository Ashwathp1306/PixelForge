import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add parent directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import PairedImageDataset
from models.builder import build_model
from utils import calculate_psnr, calculate_ssim_tensor, plot_comparison_grid, plot_intensity_histograms

def run_demo():
    print("Running PixelForge Visual Pipeline Demo...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset path
    gt_dir = "../data/sample_dataset/val/gt"
    noisy_dir = "../data/sample_dataset/val/noisy"
    
    if not (os.path.exists(gt_dir) and os.path.exists(noisy_dir)):
        gt_dir = "data/sample_dataset/val/gt"
        noisy_dir = "data/sample_dataset/val/noisy"

    dataset = PairedImageDataset(gt_dir, noisy_dir, is_train=False)
    if len(dataset) == 0:
        print("Dataset not found. Generating synthetic data first...")
        from data.generate_synthetic_data import main as gen_data
        gen_data()
        dataset = PairedImageDataset(gt_dir, noisy_dir, is_train=False)

    # Load UNet model
    model = build_model(model_name="unet", in_channels=3, out_channels=3).to(device)
    model.eval()

    # Load sample
    noisy_tensor, gt_tensor, filename = dataset[0]
    noisy = noisy_tensor.unsqueeze(0).to(device)
    gt = gt_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        restored = model(noisy)

    psnr_val = calculate_psnr(restored[0], gt[0])
    ssim_val = calculate_ssim_tensor(restored[0], gt[0])

    print(f"Sample: {filename} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

    # Convert to numpy arrays
    gt_np = gt[0].cpu().numpy().transpose(1, 2, 0)
    restored_np = restored[0].cpu().numpy().transpose(1, 2, 0)
    
    corrupt_upsampled = F.interpolate(noisy, size=(256, 256), mode='bilinear', align_corners=False)
    corrupt_np = corrupt_upsampled[0].cpu().numpy().transpose(1, 2, 0)

    # Plot & display
    os.makedirs("demo_outputs", exist_ok=True)
    plot_comparison_grid(gt_np, corrupt_np, restored_np, psnr_val=psnr_val, ssim_val=ssim_val, save_path="demo_outputs/demo_grid.png")
    plot_intensity_histograms(gt_np, corrupt_np, restored_np, save_path="demo_outputs/demo_histogram.png")

    print("Saved demo grid to 'demo_outputs/demo_grid.png'")
    print("Saved intensity histogram to 'demo_outputs/demo_histogram.png'")

if __name__ == "__main__":
    run_demo()

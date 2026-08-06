import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import PairedImageDataset
from models.builder import build_model
from utils import calculate_psnr, calculate_ssim_tensor, plot_comparison_grid, plot_intensity_histograms

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Evaluation on device: {device}")

    # Load dataset
    val_dataset = PairedImageDataset(args.gt_val_dir, args.noisy_val_dir, is_train=False, norm_type=args.norm_type)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Loaded {len(val_dataset)} validation samples from {args.gt_val_dir}")

    # Load Model
    model = build_model(model_name=args.model_name, in_channels=args.channels, out_channels=args.channels).to(device)
    
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully loaded checkpoint weights from '{args.checkpoint_path}'")
    else:
        print(f"Warning: Checkpoint '{args.checkpoint_path}' not found! Running evaluation with initialized weights.")

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    grid_dir = os.path.join(args.output_dir, "grids")
    hist_dir = os.path.join(args.output_dir, "histograms")
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    psnr_list = []
    ssim_list = []

    print("\nProcessing samples...")
    with torch.no_grad():
        for idx, (noisy, gt, fname) in enumerate(val_loader):
            noisy, gt = noisy.to(device), gt.to(device)
            filename = fname[0]
            stem = os.path.splitext(filename)[0]

            # Forward pass
            restored = model(noisy)

            # Compute PSNR & SSIM metrics for single image
            psnr_val = calculate_psnr(restored[0], gt[0])
            ssim_val = calculate_ssim_tensor(restored[0], gt[0])

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            # Prepare arrays for visualization
            gt_np = gt[0].cpu().numpy().transpose(1, 2, 0)
            restored_np = restored[0].cpu().numpy().transpose(1, 2, 0)
            
            # Upsample corrupt input (128x128 -> 256x256) via bilinear interpolation for display
            corrupt_upsampled = F.interpolate(noisy, size=(256, 256), mode='bilinear', align_corners=False)
            corrupt_np = corrupt_upsampled[0].cpu().numpy().transpose(1, 2, 0)

            # Save comparison grid plot
            grid_save_path = os.path.join(grid_dir, f"{stem}_comparison.png")
            plot_comparison_grid(gt_np, corrupt_np, restored_np, psnr_val=psnr_val, ssim_val=ssim_val, save_path=grid_save_path)

            # Save pixel intensity distribution histogram plot
            hist_save_path = os.path.join(hist_dir, f"{stem}_histogram.png")
            plot_intensity_histograms(gt_np, corrupt_np, restored_np, save_path=hist_save_path)

            if idx < 5:
                print(f" Sample [{idx+1:02d}/{len(val_dataset):02d}] {filename} -> PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n" + "="*50)
    print("           EVALUATION SUMMARY REPORT          ")
    print("="*50)
    print(f" Model Architecture: {args.model_name.upper()}")
    print(f" Total Evaluation Images: {len(val_dataset)}")
    print(f" Average PSNR: {avg_psnr:.2f} dB")
    print(f" Average SSIM: {avg_ssim:.4f}")
    print(f" Saved Comparison Grids: {grid_dir}")
    print(f" Saved Histograms: {hist_dir}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Restoration Model")
    parser.add_argument("--gt_val_dir", type=str, default="data/sample_dataset/val/gt")
    parser.add_argument("--noisy_val_dir", type=str, default="data/sample_dataset/val/noisy")
    parser.add_argument("--model_name", type=str, default="unet", choices=["unet", "esrgan"])
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_unet.pth")
    parser.add_argument("--channels", type=int, default=3, choices=[1, 3])
    parser.add_argument("--norm_type", type=str, default="none", choices=["none", "zscore", "minmax"])
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    evaluate_model(args)

if __name__ == "__main__":
    main()

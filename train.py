import os
import time
import argparse
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from data.dataset import PairedImageDataset
from models.builder import build_model
from losses.combined_loss import CompositeRestorationLoss
from utils import calculate_psnr, calculate_ssim_tensor

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_pixel = 0.0
    running_ssim = 0.0

    for noisy, gt, _ in loader:
        noisy, gt = noisy.to(device), gt.to(device)

        optimizer.zero_grad()
        output = model(noisy)
        loss, loss_dict = criterion(output, gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * noisy.size(0)
        running_pixel += loss_dict["loss_pixel"] * noisy.size(0)
        running_ssim += loss_dict["loss_ssim"] * noisy.size(0)

    total_samples = len(loader.dataset)
    return running_loss / total_samples, running_pixel / total_samples, running_ssim / total_samples

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    for noisy, gt, _ in loader:
        noisy, gt = noisy.to(device), gt.to(device)
        output = model(noisy)
        loss, _ = criterion(output, gt)

        running_loss += loss.item() * noisy.size(0)
        
        # Calculate evaluation metrics across batch samples
        for i in range(noisy.size(0)):
            psnr_val = calculate_psnr(output[i], gt[i])
            ssim_val = calculate_ssim_tensor(output[i], gt[i])
            total_psnr += psnr_val
            total_ssim += ssim_val

    total_samples = len(loader.dataset)
    val_loss = running_loss / total_samples
    val_psnr = total_psnr / total_samples
    val_ssim = total_ssim / total_samples

    return val_loss, val_psnr, val_ssim

def main():
    default_gt = "data/train/GT" if os.path.exists("data/train/GT") else "/kaggle/input/datasets/ashwath1306/dataset/train/GT"
    default_noisy = "data/train/NoisyLR" if os.path.exists("data/train/NoisyLR") else "/kaggle/input/datasets/ashwath1306/dataset/train/NoisyLR"
    default_ckpt = "checkpoints" if not os.path.exists("/kaggle") else "/kaggle/working/checkpoints"
    default_log = "runs" if not os.path.exists("/kaggle") else "/kaggle/working/runs"

    parser = argparse.ArgumentParser(description="PixelForge - PyTorch Image Restoration Training Pipeline")
    parser.add_argument("--gt_dir", type=str, default=default_gt, help="Path to GT images/npy folder")
    parser.add_argument("--noisy_dir", type=str, default=default_noisy, help="Path to Noisy LR images/npy folder")
    parser.add_argument("--gt_val_dir", type=str, default=None, help="Optional separate GT val folder")
    parser.add_argument("--noisy_val_dir", type=str, default=None, help="Optional separate Noisy LR val folder")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of dataset to use for validation if val_dir not specified")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split reproducibility")
    
    parser.add_argument("--model_name", type=str, default="unet", choices=["unet", "esrgan"], help="Model architecture")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3], help="Image channels (1 for grayscale)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    parser.add_argument("--lambda_pixel", type=float, default=1.0, help="Pixel loss weight")
    parser.add_argument("--lambda_ssim", type=float, default=0.2, help="SSIM loss weight")
    parser.add_argument("--lambda_perceptual", type=float, default=0.0, help="Perceptual loss weight")
    parser.add_argument("--use_perceptual", action="store_true", help="Enable VGG perceptual loss")
    parser.add_argument("--norm_type", type=str, default="none", choices=["none", "zscore", "minmax"], help="LR normalization type")

    parser.add_argument("--checkpoint_dir", type=str, default=default_ckpt, help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default=default_log, help="TensorBoard log directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Model: {args.model_name.upper()} | Channels: {args.channels}")

    # Datasets and Loaders setup
    if args.gt_val_dir and args.noisy_val_dir and os.path.exists(args.gt_val_dir) and os.path.exists(args.noisy_val_dir):
        train_dataset = PairedImageDataset(args.gt_dir, args.noisy_dir, is_train=True, norm_type=args.norm_type)
        val_dataset = PairedImageDataset(args.gt_val_dir, args.noisy_val_dir, is_train=False, norm_type=args.norm_type)
    else:
        full_train_ds = PairedImageDataset(args.gt_dir, args.noisy_dir, is_train=True, norm_type=args.norm_type)
        full_val_ds = PairedImageDataset(args.gt_dir, args.noisy_dir, is_train=False, norm_type=args.norm_type)
        
        total_len = len(full_train_ds)
        val_len = int(total_len * args.val_split)
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(args.seed)
        train_indices, val_indices = random_split(range(total_len), [train_len, val_len], generator=generator)
        
        train_dataset = Subset(full_train_ds, train_indices)
        val_dataset = Subset(full_val_ds, val_indices)
        print(f"Loaded {total_len} paired samples from '{args.gt_dir}'.")
        print(f"Split into {train_len} training samples and {val_len} validation samples (val_split={args.val_split}).")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Instantiate UNetSR with in_channels=1, out_channels=1 (args.channels)
    model = build_model(model_name=args.model_name, in_channels=args.channels, out_channels=args.channels).to(device)
    criterion = CompositeRestorationLoss(
        lambda_pixel=args.lambda_pixel,
        lambda_ssim=args.lambda_ssim,
        lambda_perceptual=args.lambda_perceptual,
        use_perceptual=args.use_perceptual
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Logging and Checkpointing Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"{args.model_name}_{int(time.time())}"))
    
    csv_log_path = os.path.join(args.checkpoint_dir, f"metrics_{args.model_name}.csv")
    
    # Checkpoint Auto-Detection and Resumption
    start_epoch = 1
    best_val_loss = float("inf")
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, f"latest_{args.model_name}.pth")
    
    if os.path.exists(latest_checkpoint_path):
        print(f"Found existing checkpoint at {latest_checkpoint_path}. Resuming training...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}. Best Val Loss so far: {best_val_loss:.4f}")
        
        csv_file = open(csv_log_path, mode='a', newline='')
        csv_writer = csv.writer(csv_file)
    else:
        print("No checkpoint found. Starting training from epoch 1.")
        csv_file = open(csv_log_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val PSNR (dB)", "Val SSIM", "LR"])

    end_epoch = args.epochs if args.epochs >= start_epoch else (start_epoch + args.epochs - 1)
    print(f"Training from epoch {start_epoch} to {end_epoch} (total target epochs: {end_epoch})...")
    start_time = time.time()

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_pixel, train_ssim = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # TensorBoard logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Metrics/Val_PSNR", val_psnr, epoch)
        writer.add_scalar("Metrics/Val_SSIM", val_ssim, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        # CSV logging
        csv_writer.writerow([epoch, train_loss, val_loss, val_psnr, val_ssim, current_lr])
        csv_file.flush()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

        # Save Checkpoints
        latest_path = os.path.join(args.checkpoint_dir, f"latest_{args.model_name}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": float(val_loss),
            "best_val_loss": float(best_val_loss),
            "val_psnr": float(val_psnr),
            "val_ssim": float(val_ssim)
        }, latest_path)
        print(f"Saved latest checkpoint to: {latest_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, f"best_{args.model_name}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": float(val_loss),
                "best_val_loss": float(best_val_loss),
                "val_psnr": float(val_psnr),
                "val_ssim": float(val_ssim)
            }, best_path)
            print(f"  -> Saved new best checkpoint to: {best_path} (Val Loss: {val_loss:.4f})")

    csv_file.close()
    writer.close()
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s! Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()

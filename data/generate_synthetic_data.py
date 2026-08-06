import os
import argparse
import numpy as np
from PIL import Image

def generate_synthetic_gt(h=256, w=256, c=3):
    """Generates synthetic high-resolution geometric image with rich textures and gradients."""
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    
    # Create interesting geometric and gradient patterns
    r = 0.5 + 0.5 * np.sin(10 * np.sqrt((xx - 0.5)**2 + (yy - 0.5)**2))
    g = 0.5 + 0.5 * np.cos(15 * xx)
    b = 0.5 + 0.5 * np.sin(20 * yy)
    
    if c == 1:
        img = (0.299 * r + 0.587 * g + 0.114 * b)[:, :, None]
    else:
        img = np.stack([r, g, b], axis=-1)
        
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def corrupt_image(gt_img, lr_size=(128, 128)):
    """
    Applies speckle noise, 2x down-sampling (256x256 -> 128x128), and additive Gaussian noise.
    Produces values that can naturally exceed [0, 1] range.
    """
    h_gt, w_gt, c = gt_img.shape
    h_lr, w_lr = lr_size

    # Step 1: Speckle noise degradation: y = x + x * n_speckle
    speckle_noise = np.random.normal(loc=0.0, scale=0.15, size=gt_img.shape).astype(np.float32)
    degraded = gt_img + gt_img * speckle_noise

    # Step 2: Downsampling (256 -> 128) via simple area averaging / subsampling
    step_y = h_gt // h_lr
    step_x = w_gt // w_lr
    degraded_lr = degraded[::step_y, ::step_x, :]

    # Step 3: Additive Gaussian noise (scale 0.1)
    gaussian_noise = np.random.normal(loc=0.0, scale=0.08, size=degraded_lr.shape).astype(np.float32)
    corrupt_lr = degraded_lr + gaussian_noise

    # Values in corrupt_lr will naturally exceed [0, 1] range!
    return corrupt_lr.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for testing PixelForge pipeline")
    parser.add_argument("--output_dir", type=str, default="data/sample_dataset", help="Output directory path")
    parser.add_argument("--num_train", type=int, default=32, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=8, help="Number of validation samples")
    parser.add_argument("--channels", type=int, default=3, choices=[1, 3], help="Image channels (1 for grayscale, 3 for RGB)")
    args = parser.parse_args()

    for split, count in [("train", args.num_train), ("val", args.num_val)]:
        gt_dir = os.path.join(args.output_dir, split, "gt")
        noisy_dir = os.path.join(args.output_dir, split, "noisy")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(noisy_dir, exist_ok=True)

        print(f"Generating {count} paired samples for {split} split...")
        for i in range(count):
            np.random.seed(i + (1000 if split == "val" else 0))
            gt = generate_synthetic_gt(256, 256, args.channels)
            noisy = corrupt_image(gt, (128, 128))

            # Save GT as PNG
            gt_img_uint8 = (gt * 255.0).astype(np.uint8)
            if args.channels == 1:
                Image.fromarray(gt_img_uint8.squeeze(), mode='L').save(os.path.join(gt_dir, f"sample_{i:04d}.png"))
            else:
                Image.fromarray(gt_img_uint8, mode='RGB').save(os.path.join(gt_dir, f"sample_{i:04d}.png"))

            # Save Noisy LR as .npy array to preserve raw floating values exceeding [0, 1]
            np.save(os.path.join(noisy_dir, f"sample_{i:04d}.npy"), noisy)

    print(f"Dataset successfully created in '{args.output_dir}'!")

if __name__ == "__main__":
    main()

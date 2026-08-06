import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from typing import Optional, Callable, Tuple, List, Union
from .transforms import PairedTransforms

def load_file(path: str) -> np.ndarray:
    """Load image or npy array from path into a float32 numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path).astype(np.float32)
    elif ext == '.pt':
        arr = torch.load(path).cpu().numpy().astype(np.float32)
    else:
        img = Image.open(path)
        arr = np.array(img).astype(np.float32)
        # Normalize 8-bit or 16-bit integer images to [0, 1]
        if arr.dtype == np.uint8:
            arr /= 255.0
        elif arr.dtype == np.uint16:
            arr /= 65535.0
            
    # Ensure shape (H, W, C)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr

class PairedImageDataset(data.Dataset):
    """
    Dataset class for paired Noisy LR (128x128) and GT (256x256) images.
    
    Supports:
    - Custom directory paths or single directory with subfolders
    - Image extensions (.png, .jpg, .tif, .npy, .pt)
    - Out-of-bounds intensity values in Noisy LR images
    - Synchronized paired spatial augmentations
    """
    def __init__(
        self,
        gt_dir: str,
        noisy_dir: str,
        transform: Optional[Callable] = None,
        is_train: bool = True,
        norm_type: str = "none"  # "none", "zscore", or "minmax"
    ):
        super().__init__()
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.transform = transform if transform is not None else PairedTransforms(is_train=is_train)
        self.is_train = is_train
        self.norm_type = norm_type

        # Find matching paired file paths
        supported_exts = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.npy', '*.pt']
        self.gt_paths = []
        self.noisy_paths = []

        if os.path.exists(gt_dir) and os.path.exists(noisy_dir):
            for ext in supported_exts:
                self.gt_paths.extend(glob.glob(os.path.join(gt_dir, ext)))
                self.gt_paths.extend(glob.glob(os.path.join(gt_dir, ext.upper())))

            # Deduplicate: on Windows, matching both lowercase and uppercase
            # extension patterns can return the same file twice (Windows
            # filenames aren't case-sensitive), sometimes with different
            # path separators too. Normalize and dedupe before sorting.
            self.gt_paths = sorted(set(os.path.normpath(p) for p in self.gt_paths))

            for gt_p in self.gt_paths:
                fname = os.path.basename(gt_p)
                noisy_p = os.path.join(noisy_dir, fname)
                if not os.path.exists(noisy_p):
                    # Fallback to searching matching stem if extension differs
                    stem = os.path.splitext(fname)[0]
                    matches = glob.glob(os.path.join(noisy_dir, f"{stem}.*"))
                    if len(matches) > 0:
                        noisy_p = matches[0]
                    else:
                        raise FileNotFoundError(f"Could not find matching noisy image for GT: {gt_p} in {noisy_dir}")
                self.noisy_paths.append(noisy_p)
        else:
            print(f"Warning: Directory {gt_dir} or {noisy_dir} does not exist yet. Dataset initialized as empty.")

    def __len__(self) -> int:
        return len(self.gt_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        gt_path = self.gt_paths[index]
        noisy_path = self.noisy_paths[index]

        # Load raw numpy arrays (H, W, C)
        gt_arr = load_file(gt_path)
        noisy_arr = load_file(noisy_path)

        # Apply normalization handling to Noisy LR if specified
        if self.norm_type == "zscore":
            mean, std = noisy_arr.mean(), noisy_arr.std() + 1e-6
            noisy_arr = (noisy_arr - mean) / std
        elif self.norm_type == "minmax":
            min_v, max_v = noisy_arr.min(), noisy_arr.max()
            if max_v > min_v:
                noisy_arr = (noisy_arr - min_v) / (max_v - min_v)

        # Convert to PyTorch Tensors (C, H, W)
        gt_tensor = torch.from_numpy(gt_arr).permute(2, 0, 1).float()
        noisy_tensor = torch.from_numpy(noisy_arr).permute(2, 0, 1).float()

        # Apply paired transforms
        noisy_tensor, gt_tensor = self.transform(noisy_tensor, gt_tensor)

        return noisy_tensor, gt_tensor, os.path.basename(gt_path)
import torch
import random
import torchvision.transforms.functional as TF

class PairedTransforms:
    """
    Synchronized spatial data augmentations for paired NoisyLR and GT images.
    Applies random horizontal flip, vertical flip, and 90-degree rotations.
    """
    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def __call__(self, noisy_lr: torch.Tensor, gt: torch.Tensor):
        """
        noisy_lr: Tensor (C, 128, 128)
        gt: Tensor (C, 256, 256)
        """
        if not self.is_train:
            return noisy_lr, gt

        # Random Horizontal Flip
        if random.random() > 0.5:
            noisy_lr = TF.hflip(noisy_lr)
            gt = TF.hflip(gt)

        # Random Vertical Flip
        if random.random() > 0.5:
            noisy_lr = TF.vflip(noisy_lr)
            gt = TF.vflip(gt)

        # Random 90-degree Rotation
        rot_choices = [0, 90, 180, 270]
        angle = random.choice(rot_choices)
        if angle != 0:
            noisy_lr = TF.rotate(noisy_lr, angle)
            gt = TF.rotate(gt, angle)

        return noisy_lr, gt

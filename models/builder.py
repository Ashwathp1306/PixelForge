import torch.nn as nn
from .unet import UNetSR
from .esrgan import LightweightESRGAN

def build_model(
    model_name: str = "unet",
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 64
) -> nn.Module:
    """
    Factory function to instantiate image restoration / super-resolution models.
    
    Supported model_name options:
    - 'unet': ResUNet with Channel Attention and 2x PixelShuffle upsampler
    - 'esrgan': Lightweight RRDB ESRGAN Generator with 2x PixelShuffle upsampler
    """
    name = model_name.lower()
    if name == "unet":
        return UNetSR(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    elif name in ["esrgan", "rrdb"]:
        return LightweightESRGAN(in_channels=in_channels, out_channels=out_channels, num_feat=base_channels)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose between 'unet' and 'esrgan'.")

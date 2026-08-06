import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    """Dense block used in ESRGAN RRDB."""
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB)."""
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = DenseBlock(num_feat, num_grow_ch)
        self.rdb2 = DenseBlock(num_feat, num_grow_ch)
        self.rdb3 = DenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class LightweightESRGAN(nn.Module):
    """
    Lightweight ESRGAN Generator (RRDBNet) for Denoising + 2x Super Resolution.
    
    Inputs: (B, in_channels, 128, 128)
    Outputs: (B, out_channels, 256, 256)
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_feat: int = 64, num_block: int = 4, num_grow_ch: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # First Feature Extraction Layer
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)

        # RRDB Trunk
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 2x Upsampling Module (Sub-pixel Convolution)
        self.conv_up = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.up_shuffle = nn.PixelShuffle(upscale_factor=2)
        
        # Output Reconstruction
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_base = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # 2x Super-Resolution Upsampling
        feat = self.lrelu(self.up_shuffle(self.conv_up(feat)))
        out = self.conv_last(self.lrelu(self.conv_hr(feat))) + residual_base
        
        return torch.clamp(out, 0.0, 1.0)

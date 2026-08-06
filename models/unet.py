import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel Attention Block (SE-style) to focus on noise-free feature channels."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """Residual Block with LeakyReLU and Channel Attention."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = ChannelAttention(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        return self.act(out + res)

class UNetSR(nn.Module):
    """
    UNet Architecture with 2x Super-Resolution sub-pixel upsampling head.
    
    Inputs: (B, in_channels, 128, 128)
    Outputs: (B, out_channels, 256, 256)
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initial Feature Extraction
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (128x128 -> 64x64 -> 32x32 -> 16x16)
        self.enc1 = ResBlock(base_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 64x64

        self.enc2 = ResBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)  # 32x32

        self.enc3 = ResBlock(base_channels * 4, base_channels * 4)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1)  # 16x16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(base_channels * 8, base_channels * 8),
            ResBlock(base_channels * 8, base_channels * 8)
        )

        # Decoder with Skip Connections (16x16 -> 32x32 -> 64x64 -> 128x128)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(base_channels * 2, base_channels)

        # 2x Super-Resolution Sub-Pixel Upsampling Head (128x128 -> 256x256)
        self.sr_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),  # (B, base_channels, 256, 256)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save upsampled input for global residual learning
        residual_base = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)

        # Encoder
        x1 = self.head(x)
        e1 = self.enc1(x1)
        
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        # Bottleneck
        b = self.bottleneck(self.down3(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # 2x Super-Resolution
        restored = self.sr_head(d1) + residual_base
        return torch.clamp(restored, 0.0, 1.0)

"""
Masked Autoencoder — Enhanced UNet with Attention Gates.

Architecture (128×128 input)
────────────────────────────
    Input  (3 × 128 × 128)  — masked image (some patches zeroed)
    ├── Encoder 1 → 64 ch,  128×128
    │   └── MaxPool → 64×64
    ├── Encoder 2 → 128 ch, 64×64
    │   └── MaxPool → 32×32
    ├── Encoder 3 → 256 ch, 32×32
    │   └── MaxPool → 16×16
    ├── Encoder 4 → 512 ch, 16×16
    │   └── MaxPool → 8×8
    ├── Bottleneck → 1024 ch, 8×8 (with Dropout + Residual)
    ├── Decoder 4 → 512 ch, 16×16 (+ attention-gated skip from Enc 4)
    ├── Decoder 3 → 256 ch, 32×32 (+ attention-gated skip from Enc 3)
    ├── Decoder 2 → 128 ch, 64×64 (+ attention-gated skip from Enc 2)
    ├── Decoder 1 →  64 ch, 128×128 (+ attention-gated skip from Enc 1)
    └── Output   →   3 ch, 128×128 (Sigmoid → [0, 1])

Key features:
  • 4-level encoder/decoder with skip connections
  • Attention gates on skip connections for selective feature fusion
  • Residual connections in bottleneck
  • Dropout for regularization
  • Refined output head for sharper reconstructions
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution: Conv3×3 → BN → ReLU  ×2, with optional residual."""

    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.
    Learns to focus on relevant spatial regions from encoder features.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attention = self.psi(self.relu(g + s))
        return skip * attention


class MaskedAutoencoder(nn.Module):
    """
    UNet-style autoencoder with attention gates for self-supervised
    masked image reconstruction.

    The model receives a *masked* image (patches set to 0) and learns to
    reconstruct the full, unmasked image. Attention-gated skip connections
    help the decoder focus on relevant encoder features.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, 64, use_residual=False)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128, use_residual=False)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256, use_residual=False)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(256, 512, use_residual=False)
        self.pool4 = nn.MaxPool2d(2)

        # ── Bottleneck ───────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 1024, use_residual=False),
            nn.Dropout2d(0.2),
            ConvBlock(1024, 1024, use_residual=True),
        )

        # ── Decoder (with attention-gated skip connections) ──────────────
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(gate_channels=512, skip_channels=512, inter_channels=256)
        self.dec4 = ConvBlock(1024, 512, use_residual=True)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(gate_channels=256, skip_channels=256, inter_channels=128)
        self.dec3 = ConvBlock(512, 256, use_residual=True)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(gate_channels=128, skip_channels=128, inter_channels=64)
        self.dec2 = ConvBlock(256, 128, use_residual=True)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(gate_channels=64, skip_channels=64, inter_channels=32)
        self.dec1 = ConvBlock(128, 64, use_residual=True)

        # ── Output head ─────────────────────────────────────────────────
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.Sigmoid(),  # pixel values in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W) masked input image.
        Returns:
            Reconstructed image of the same shape.
        """
        # Encoder
        e1 = self.enc1(x)                            # (B,   64, 128, 128)
        e2 = self.enc2(self.pool1(e1))               # (B,  128,  64,  64)
        e3 = self.enc3(self.pool2(e2))               # (B,  256,  32,  32)
        e4 = self.enc4(self.pool3(e3))               # (B,  512,  16,  16)

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))          # (B, 1024,   8,   8)

        # Decoder with attention gates
        d4 = self.up4(b)                             # (B,  512,  16,  16)
        e4_att = self.attn4(gate=d4, skip=e4)
        d4 = self.dec4(torch.cat([d4, e4_att], 1))   # (B,  512,  16,  16)

        d3 = self.up3(d4)                            # (B,  256,  32,  32)
        e3_att = self.attn3(gate=d3, skip=e3)
        d3 = self.dec3(torch.cat([d3, e3_att], 1))   # (B,  256,  32,  32)

        d2 = self.up2(d3)                            # (B,  128,  64,  64)
        e2_att = self.attn2(gate=d2, skip=e2)
        d2 = self.dec2(torch.cat([d2, e2_att], 1))   # (B,  128,  64,  64)

        d1 = self.up1(d2)                            # (B,   64, 128, 128)
        e1_att = self.attn1(gate=d1, skip=e1)
        d1 = self.dec1(torch.cat([d1, e1_att], 1))   # (B,   64, 128, 128)

        return self.output_conv(d1)                  # (B,    3, 128, 128)

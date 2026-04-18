"""
══════════════════════════════════════════════════════════════════════════════
  Google Colab Training Script — Masked Autoencoder (Self-Contained)
══════════════════════════════════════════════════════════════════════════════

HOW TO USE (Google Colab — FREE GPU):
─────────────────────────────────────
  OPTION A — Run this single file:
    1. Go to https://colab.research.google.com
    2. Click Runtime → Change runtime type → GPU (T4)
    3. Upload THIS FILE to Colab (or paste into a cell)
    4. Run it. That's it!

  OPTION B — Clone from GitHub:
    1. Open Colab → New notebook
    2. Run: !git clone https://github.com/YOUR_USERNAME/MMProject.git
    3. Run: %cd MMProject
    4. Run: !python colab_train.py
    5. Download: checkpoints/best_model.pth

WHAT IT DOES:
  • Installs dependencies
  • Downloads STL-10 dataset (96×96 images — much better than CIFAR-10's 32×32)
  • Trains the Masked Autoencoder with attention gates for 100 epochs
  • Uses combined loss: MSE + L1 + SSIM → sharp, high-quality reconstructions
  • Saves best_model.pth + training history plot
  • Shows sample reconstructions

ESTIMATED TIME: ~30-40 minutes on T4 GPU (free Colab)
══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
#  CELL 1: Setup & GPU Check
# ═══════════════════════════════════════════════════════════════════════════

import subprocess, sys, os

# Install dependencies
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'torch', 'torchvision', 'scikit-image', 'matplotlib', 'Pillow'])

print("✅ Dependencies installed")

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("   ⚠️  No GPU detected! Training will be much slower.")
    print("   → Go to Runtime → Change runtime type → GPU")


# ═══════════════════════════════════════════════════════════════════════════
#  CELL 2: Configuration
# ═══════════════════════════════════════════════════════════════════════════

# ─── Hyperparameters (adjust these if needed) ────────────────────────────
IMG_SIZE    = 128      # Resolution (128×128 — 4× more pixels than old 64×64)
PATCH_SIZE  = 8        # Each mask patch covers 8×8 pixels (256 total patches)
EPOCHS      = 100      # Training epochs (100 is good, 50 for quick test)
BATCH_SIZE  = 64       # Batch size (64 works well on T4, reduce to 32 if OOM)
LR          = 5e-4     # Learning rate
MASK_RATIO  = 0.5      # Fraction of patches to mask
SAVE_DIR    = './checkpoints'
DATA_DIR    = './data'

os.makedirs(SAVE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CELL 3: Model Architecture (MUST match model/model.py exactly!)
# ═══════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


# ── Masking ──────────────────────────────────────────────────────────────

def create_patch_mask(image_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=0.5):
    """Create a random patch-based binary mask."""
    patches_per_side = image_size // patch_size
    num_patches = patches_per_side ** 2
    num_masked = int(num_patches * mask_ratio)
    perm = torch.randperm(num_patches)
    masked_indices = perm[:num_masked]
    mask = torch.ones(num_patches)
    mask[masked_indices] = 0.0
    mask = mask.reshape(patches_per_side, patches_per_side)
    mask = mask.repeat_interleave(patch_size, dim=0)
    mask = mask.repeat_interleave(patch_size, dim=1)
    return mask.unsqueeze(0), masked_indices


# ── SSIM Loss ────────────────────────────────────────────────────────────

def ssim_loss_torch(pred, target, window_size=11):
    """Differentiable SSIM loss: returns 1 - SSIM."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(pred.size(1), 1, -1, -1).contiguous()
    pad = window_size // 2

    mu1 = torch.nn.functional.conv2d(pred, window, padding=pad, groups=pred.size(1))
    mu2 = torch.nn.functional.conv2d(target, window, padding=pad, groups=pred.size(1))
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(pred * pred, window, padding=pad, groups=pred.size(1)) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, window, padding=pad, groups=pred.size(1)) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * target, window, padding=pad, groups=pred.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


# ── Model Blocks ─────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Double convolution: Conv3×3 → BN → ReLU ×2, with optional residual."""
    def __init__(self, in_channels, out_channels, use_residual=False):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_residual else out


class AttentionGate(nn.Module):
    """Attention gate for skip connections — focuses on relevant features."""
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        attention = self.psi(self.relu(self.W_gate(gate) + self.W_skip(skip)))
        return skip * attention


class MaskedAutoencoder(nn.Module):
    """
    UNet-style autoencoder with attention gates.
    Input: (B, 3, 128, 128) masked image
    Output: (B, 3, 128, 128) reconstructed image

    ⚠️  This architecture MUST match model/model.py exactly!
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64);  self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128);          self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256);         self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512);         self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 1024),
            nn.Dropout2d(0.2),
            ConvBlock(1024, 1024, use_residual=True),
        )

        # Decoder with attention gates
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.attn4 = AttentionGate(512, 512, 256)
        self.dec4 = ConvBlock(1024, 512, use_residual=True)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.attn3 = AttentionGate(256, 256, 128)
        self.dec3 = ConvBlock(512, 256, use_residual=True)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.attn2 = AttentionGate(128, 128, 64)
        self.dec2 = ConvBlock(256, 128, use_residual=True)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.attn1 = AttentionGate(64, 64, 32)
        self.dec1 = ConvBlock(128, 64, use_residual=True)

        # Output head
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Decoder with attention-gated skips
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self.attn4(d4, e4)], 1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.attn3(d3, e3)], 1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.attn2(d2, e2)], 1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.attn1(d1, e1)], 1))

        return self.output_conv(d1)


# ═══════════════════════════════════════════════════════════════════════════
#  CELL 4: Dataset — STL-10 (96×96 native, much better than CIFAR-10)
# ═══════════════════════════════════════════════════════════════════════════

class MaskedImageDataset(Dataset):
    def __init__(self, base_dataset, mask_ratio=0.5):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio
    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        image = item[0] if isinstance(item, (tuple, list)) else item
        mask, _ = create_patch_mask(mask_ratio=self.mask_ratio)
        return image * mask, image, mask


# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# STL-10: train (5K) + test (8K) = 13K images at 96×96
print("\n📦 Downloading STL-10 dataset...")
train_ds = datasets.STL10(root=DATA_DIR, split='train', download=True, transform=train_transform)
val_ds = datasets.STL10(root=DATA_DIR, split='test', download=True, transform=val_transform)

train_loader = DataLoader(MaskedImageDataset(train_ds, MASK_RATIO),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(MaskedImageDataset(val_ds, MASK_RATIO),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"✅ Dataset loaded: {len(train_ds)} train / {len(val_ds)} val images")
print(f"   Image size: {IMG_SIZE}×{IMG_SIZE} (upscaled from 96×96 STL-10)")
print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


# ═══════════════════════════════════════════════════════════════════════════
#  CELL 5: Training Loop
# ═══════════════════════════════════════════════════════════════════════════

model = MaskedAutoencoder().to(device)
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

num_params = sum(p.numel() for p in model.parameters())
print(f"\n{'='*74}")
print(f"  Model parameters: {num_params:,}")
print(f"  Training {EPOCHS} epochs on {device}")
print(f"  Loss: 0.5×MSE + 0.3×L1 + 0.2×SSIM (sharp reconstructions)")
print(f"{'='*74}\n")

train_losses, val_losses = [], []
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train ────────────────────────────────────────────────────────────
    model.train()
    running = 0.0
    for masked, original, mask in train_loader:
        masked, original = masked.to(device), original.to(device)
        optimizer.zero_grad()
        out = model(masked)

        # Combined loss for sharp reconstructions
        loss_mse = mse_criterion(out, original)
        loss_l1 = l1_criterion(out, original)
        loss_ssim = ssim_loss_torch(out, original)
        loss = 0.5 * loss_mse + 0.3 * loss_l1 + 0.2 * loss_ssim

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item()
    avg_train = running / len(train_loader)
    train_losses.append(avg_train)

    # ── Validate ─────────────────────────────────────────────────────────
    model.eval()
    running = 0.0
    with torch.no_grad():
        for masked, original, mask in val_loader:
            masked, original = masked.to(device), original.to(device)
            out = model(masked)
            loss_mse = mse_criterion(out, original)
            loss_l1 = l1_criterion(out, original)
            loss_ssim = ssim_loss_torch(out, original)
            loss = 0.5 * loss_mse + 0.3 * loss_l1 + 0.2 * loss_ssim
            running += loss.item()
    avg_val = running / len(val_loader)
    val_losses.append(avg_val)

    scheduler.step()
    elapsed = time.time() - t0

    marker = ""
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train, 'val_loss': avg_val,
            'img_size': IMG_SIZE,
        }, os.path.join(SAVE_DIR, 'best_model.pth'))
        marker = " ★"

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  │  train {avg_train:.6f}  │  "
              f"val {avg_val:.6f}  │  {elapsed:.1f}s{marker}")

    # Save periodic checkpoint
    if epoch % 25 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train, 'val_loss': avg_val,
            'img_size': IMG_SIZE,
        }, os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch}.pth'))

print(f"\n✅ Training complete! Best val loss: {best_val_loss:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  CELL 6: Plot Training History
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, 'b-', lw=2, label='Train')
ax.plot(val_losses, 'r-', lw=2, label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('Combined Loss (MSE+L1+SSIM)')
ax.set_title('Training History — Masked Autoencoder')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'), dpi=150)
plt.show()
print("✅ History plot saved")


# ═══════════════════════════════════════════════════════════════════════════
#  CELL 7: Sample Reconstructions (Visual Check)
# ═══════════════════════════════════════════════════════════════════════════

model.eval()
sample_batch = next(iter(val_loader))
masked_imgs, originals, masks = sample_batch
masked_imgs = masked_imgs.to(device)

with torch.no_grad():
    reconstructed = model(masked_imgs).cpu()

# Show 4 samples
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
for i in range(4):
    # Original
    axes[i, 0].imshow(originals[i].permute(1, 2, 0).numpy().clip(0, 1))
    axes[i, 0].set_title('Original' if i == 0 else '')
    axes[i, 0].axis('off')

    # Masked
    axes[i, 1].imshow(masked_imgs[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
    axes[i, 1].set_title('Masked Input' if i == 0 else '')
    axes[i, 1].axis('off')

    # Reconstructed
    axes[i, 2].imshow(reconstructed[i].permute(1, 2, 0).numpy().clip(0, 1))
    axes[i, 2].set_title('Reconstructed' if i == 0 else '')
    axes[i, 2].axis('off')

plt.suptitle('Sample Reconstructions (128×128)', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'sample_reconstructions.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sample reconstructions saved")


# ═══════════════════════════════════════════════════════════════════════════
#  CELL 8: Download Checkpoint
# ═══════════════════════════════════════════════════════════════════════════

# Uncomment these lines in Google Colab to download the trained model:

# from google.colab import files
# files.download('checkpoints/best_model.pth')

checkpoint_path = os.path.join(SAVE_DIR, 'best_model.pth')
checkpoint_size = os.path.getsize(checkpoint_path) / 1e6

print(f"\n{'='*74}")
print(f"  ✅ TRAINING COMPLETE!")
print(f"{'='*74}")
print(f"  📁 Checkpoint: {checkpoint_path} ({checkpoint_size:.1f} MB)")
print(f"  📊 Best val loss: {best_val_loss:.6f}")
print(f"  🖼️  Resolution: {IMG_SIZE}×{IMG_SIZE}")
print(f"")
print(f"  NEXT STEPS:")
print(f"  ─────────────────────────────────────────────────────")
print(f"  1. Download: checkpoints/best_model.pth")
print(f"     → In Colab: from google.colab import files")
print(f"                  files.download('checkpoints/best_model.pth')")
print(f"  2. Place it in your local project folder:")
print(f"     → MMProject/checkpoints/best_model.pth")
print(f"  3. Run the web app:  python app.py")
print(f"  4. Open browser:    http://localhost:5000")
print(f"{'='*74}")

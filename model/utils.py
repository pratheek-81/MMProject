"""
Utility functions for masking, metrics, and visualization.

Provides:
    - Patch-based random masking
    - PSNR and SSIM calculation
    - Error heatmap generation
    - Tensor/image conversion helpers
"""

import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────────────
IMG_SIZE = 128       # Input image resolution (128×128) — was 64, now 4× more pixels
PATCH_SIZE = 8       # Each patch is 8×8 pixels → 16×16 = 256 total patches
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 256 total patches


# ─── Masking ─────────────────────────────────────────────────────────────────

def create_patch_mask(image_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=0.5):
    """
    Create a random patch-based binary mask.

    Divides the image into a grid of (image_size/patch_size)^2 patches
    and randomly zeros out `mask_ratio` fraction of them.

    Args:
        image_size  : Spatial size of the square image (pixels).
        patch_size  : Spatial size of each square patch (pixels).
        mask_ratio  : Fraction of patches to mask (0.0–1.0).

    Returns:
        mask           : Float tensor of shape (1, H, W). 1 = visible, 0 = masked.
        masked_indices : 1-D tensor with the linear indices of masked patches.
    """
    patches_per_side = image_size // patch_size
    num_patches = patches_per_side ** 2
    num_masked = int(num_patches * mask_ratio)

    # Random permutation → first `num_masked` indices are masked
    perm = torch.randperm(num_patches)
    masked_indices = perm[:num_masked]

    # Build patch-level mask and expand to pixel-level
    mask = torch.ones(num_patches)
    mask[masked_indices] = 0.0
    mask = mask.reshape(patches_per_side, patches_per_side)
    mask = mask.repeat_interleave(patch_size, dim=0)  # expand rows
    mask = mask.repeat_interleave(patch_size, dim=1)  # expand cols
    mask = mask.unsqueeze(0)  # (1, H, W)

    return mask, masked_indices


def apply_mask(image, mask):
    """
    Element-wise multiply an image by a binary mask.

    Args:
        image : Tensor (C, H, W) or (B, C, H, W).
        mask  : Tensor (1, H, W) or (B, 1, H, W).

    Returns:
        Masked image (same shape as input).
    """
    return image * mask


# ─── Metrics ─────────────────────────────────────────────────────────────────

def calculate_psnr(original, reconstructed):
    """
    Peak Signal-to-Noise Ratio between two images.

    Args:
        original      : ndarray (H, W, C) in [0, 1].
        reconstructed : ndarray (H, W, C) in [0, 1].

    Returns:
        PSNR in dB (float). Returns inf when images are identical.
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def calculate_ssim(original, reconstructed):
    """
    Structural Similarity Index between two images.

    Args:
        original      : ndarray (H, W, C) in [0, 1].
        reconstructed : ndarray (H, W, C) in [0, 1].

    Returns:
        SSIM value (float in [-1, 1], higher is better).
    """
    # For 128×128 images, win_size=7 works well
    win_size = min(7, min(original.shape[0], original.shape[1]))
    if win_size % 2 == 0:
        win_size -= 1  # must be odd
    win_size = max(win_size, 3)

    return ssim_func(
        original,
        reconstructed,
        channel_axis=2,
        data_range=1.0,
        win_size=win_size,
    )


# ─── SSIM Loss (for training) ───────────────────────────────────────────────

def ssim_loss_torch(pred, target, window_size=11):
    """
    Differentiable SSIM loss for PyTorch training.
    Returns 1 - SSIM (so lower is better, like MSE).

    Args:
        pred   : Tensor (B, C, H, W) in [0, 1].
        target : Tensor (B, C, H, W) in [0, 1].

    Returns:
        Scalar loss tensor.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)  # 2D Gaussian
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    window = window.expand(pred.size(1), 1, -1, -1).contiguous()

    pad = window_size // 2

    mu1 = torch.nn.functional.conv2d(pred, window, padding=pad, groups=pred.size(1))
    mu2 = torch.nn.functional.conv2d(target, window, padding=pad, groups=pred.size(1))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(pred * pred, window, padding=pad, groups=pred.size(1)) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, window, padding=pad, groups=pred.size(1)) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * target, window, padding=pad, groups=pred.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


# ─── Visualization helpers ──────────────────────────────────────────────────

def generate_error_heatmap(original, reconstructed, save_path):
    """
    Compute a per-pixel absolute-error map and save it as a 'hot' heatmap.

    Args:
        original      : ndarray (H, W, C) in [0, 1].
        reconstructed : ndarray (H, W, C) in [0, 1].
        save_path     : File path for the output PNG.
    """
    error = np.mean(np.abs(original - reconstructed), axis=2)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.5)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150, transparent=False,
                facecolor='#0b0d17')
    plt.close(fig)


def tensor_to_numpy(tensor):
    """
    Convert a (C, H, W) or (1, C, H, W) tensor to (H, W, C) ndarray in [0, 1].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def save_tensor_as_image(tensor, save_path):
    """Save a tensor as a high-quality PNG image using PIL (no compression artifacts)."""
    img_np = tensor_to_numpy(tensor)
    # Convert [0, 1] float to [0, 255] uint8
    img_uint8 = (img_np * 255).round().astype(np.uint8)
    Image.fromarray(img_uint8).save(save_path, format='PNG')

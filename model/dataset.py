"""
Dataset module — STL-10 and custom image-folder loaders.

STL-10 is preferred over CIFAR-10 because:
  • Native resolution: 96×96 (vs 32×32) — much closer to our 128×128 target
  • Includes 100K unlabeled images — ideal for self-supervised learning
  • Richer, more diverse images → better generalization

Both loaders return (masked_image, original_image, mask) triples
via the `MaskedImageDataset` wrapper.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image

from .utils import IMG_SIZE, create_patch_mask, apply_mask


# ─── Masked wrapper ─────────────────────────────────────────────────────────

class MaskedImageDataset(Dataset):
    """
    Wraps any image dataset and adds random patch masking on-the-fly.

    Each call to __getitem__ returns:
        masked_image : Tensor (C, H, W)  — image with some patches zeroed
        original     : Tensor (C, H, W)  — clean target
        mask         : Tensor (1, H, W)  — binary mask (1 = visible)
    """

    def __init__(self, base_dataset, mask_ratio: float = 0.5):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        image = item[0] if isinstance(item, (tuple, list)) else item

        mask, _ = create_patch_mask(mask_ratio=self.mask_ratio)
        masked_image = apply_mask(image, mask)

        return masked_image, image, mask


# ─── Data augmentation transforms ───────────────────────────────────────────

def get_train_transform():
    """Training transform with augmentation for better generalization."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])


def get_val_transform():
    """Validation/inference transform — no augmentation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


# ─── STL-10 loader (recommended — 96×96 native) ────────────────────────────

def get_stl10_dataset(data_dir: str = './data',
                      mask_ratio: float = 0.5,
                      batch_size: int = 64,
                      use_unlabeled: bool = False,
                      num_workers: int = 0):
    """
    Download STL-10 and return masked data loaders.

    STL-10 has:
      • train split: 5,000 images (96×96)
      • test split:  8,000 images (96×96)
      • unlabeled:  100,000 images (96×96) — optional, for longer training

    Args:
        data_dir      : Where to cache the raw STL-10 files.
        mask_ratio    : Fraction of patches to mask (0.0–1.0).
        batch_size    : Mini-batch size.
        use_unlabeled : If True, also include the 100K unlabeled images.
        num_workers   : DataLoader workers (0 is safest on Windows).

    Returns:
        train_loader, val_loader  (DataLoader instances)
    """
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    # Training data: train split (+ optionally unlabeled split)
    train_ds = datasets.STL10(
        root=data_dir, split='train', download=True, transform=train_transform,
    )

    if use_unlabeled:
        unlabeled_ds = datasets.STL10(
            root=data_dir, split='unlabeled', download=True, transform=train_transform,
        )
        train_ds = ConcatDataset([train_ds, unlabeled_ds])
        print(f"[INFO] Using STL-10 train + unlabeled: {len(train_ds)} images")
    else:
        print(f"[INFO] Using STL-10 train split: {len(train_ds)} images")

    # Validation data: test split
    val_ds = datasets.STL10(
        root=data_dir, split='test', download=True, transform=val_transform,
    )
    print(f"[INFO] Using STL-10 test split for validation: {len(val_ds)} images")

    train_loader = DataLoader(
        MaskedImageDataset(train_ds, mask_ratio),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        MaskedImageDataset(val_ds, mask_ratio),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ─── CIFAR-10 loader (fallback — smaller dataset) ──────────────────────────

def get_cifar10_dataset(data_dir: str = './data',
                        mask_ratio: float = 0.5,
                        batch_size: int = 64,
                        subset_size: int = 0):
    """
    Download CIFAR-10, resize to IMG_SIZE, and return masked data loaders.

    Note: CIFAR-10 is only 32×32 natively. Images will be upscaled to
    IMG_SIZE which can introduce blur. Prefer STL-10 instead.

    Args:
        data_dir    : Where to cache the raw CIFAR-10 files.
        mask_ratio  : Fraction of patches to mask (0.0–1.0).
        batch_size  : Mini-batch size.
        subset_size : If > 0, use only this many images (for fast CPU training).
                      Set to 0 to use the full dataset.

    Returns:
        train_loader, val_loader  (DataLoader instances)
    """
    transform = get_val_transform()

    full_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Optionally take a small subset for fast CPU training
    if subset_size > 0 and subset_size < len(full_dataset):
        full_dataset = torch.utils.data.Subset(
            full_dataset,
            torch.randperm(len(full_dataset),
                           generator=torch.Generator().manual_seed(42))[:subset_size].tolist()
        )
        print(f"[INFO] Using subset: {subset_size} images")

    # 90 / 10 train-val split
    total = len(full_dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        MaskedImageDataset(train_ds, mask_ratio),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        MaskedImageDataset(val_ds, mask_ratio),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader


# ─── Custom image-folder loader ─────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    """
    Load all images from a flat directory.

    Supported extensions: .jpg .jpeg .png .bmp .tiff .webp
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform or get_val_transform()

        self.image_paths = sorted(
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

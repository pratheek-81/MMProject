# Self-Supervised Mask-Based Image Reconstruction with Consistency Analysis

A complete end-to-end deep learning project that uses a **self-supervised Masked Autoencoder** to reconstruct hidden regions of images and analyse reconstruction quality via PSNR, SSIM, and pixel-wise error heatmaps.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)

---

## Features

| Feature | Description |
|---------|-------------|
| **Masked Autoencoder** | UNet-style CNN with **attention gates** and skip connections (~11M params) |
| **128×128 Resolution** | 4× more pixels than previous version — dramatically sharper output |
| **Self-Supervised Training** | Trained on **STL-10** (96×96 native) — no manual labels needed |
| **Combined Loss** | MSE + L1 + SSIM → sharp, structurally accurate reconstructions |
| **Smart Blending** | Keeps original pixels in visible regions + model output for masked regions |
| **Configurable Masking** | Random patch masking with adjustable ratio (10%–90%) |
| **Consistency Analysis** | PSNR, SSIM, and per-pixel error heatmap |
| **Web Interface** | Drag-and-drop upload, live results, dark-themed glassmorphism UI |

---

## Project Structure

```
MMProject/
├── app.py                  # Flask web application
├── colab_train.py          # Self-contained Colab training script
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── model/
│   ├── __init__.py
│   ├── model.py            # UNet with Attention Gates architecture
│   ├── dataset.py          # STL-10 + CIFAR-10 + custom folder loaders
│   ├── train.py            # Training script with CLI
│   └── utils.py            # Masking, SSIM loss, metrics, visualisation
│
├── templates/
│   └── index.html          # Web UI
│
├── static/
│   ├── uploads/            # User-uploaded images
│   └── outputs/            # Generated result images
│
├── checkpoints/            # Saved model weights
└── data/                   # Dataset cache
```

---

## 🚀 Complete Guide: Train on Google Colab & Run Locally

### Step 1: Push to GitHub

```bash
# In your local MMProject folder
git add -A
git commit -m "Updated model with attention gates & 128x128 resolution"
git push origin main
```

### Step 2: Train on Google Colab (FREE GPU)

1. Go to [Google Colab](https://colab.research.google.com)
2. Create a **New Notebook**
3. **Change runtime to GPU**: `Runtime → Change runtime type → T4 GPU`
4. Run these cells:

**Cell 1 — Clone your repo:**
```python
!git clone https://github.com/YOUR_USERNAME/MMProject.git
%cd MMProject
```

**Cell 2 — Run training:**
```python
!python colab_train.py
```

This will:
- Install all dependencies automatically
- Download STL-10 dataset (~2.6 GB, one-time)
- Train for 100 epochs (~30-40 minutes on T4)
- Save `checkpoints/best_model.pth`
- Show sample reconstructions

**Cell 3 — Download the trained model:**
```python
from google.colab import files
files.download('checkpoints/best_model.pth')
```

### Step 3: Run Locally

```bash
# Place the downloaded best_model.pth in your local project
# MMProject/checkpoints/best_model.pth

# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py

# Open browser
# http://localhost:5000
```

### Alternative: Run colab_train.py directly (single cell)

If you prefer, you can also copy-paste the entire `colab_train.py` file into a single Colab cell. It's fully self-contained — no imports from other files needed.

---

## How It Works

### Self-Supervised Learning Pipeline

```
Input Image → Resize 128×128 → Random Patch Masking → UNet + Attention → Reconstructed Image
      ↑                                                                          ↓
      └──────── Combined Loss (0.5×MSE + 0.3×L1 + 0.2×SSIM) ←─────────────────┘
```

1. **Masking**: The image is divided into 8×8 pixel patches (256 total). A configurable fraction is randomly zeroed out.
2. **Reconstruction**: The UNet autoencoder with attention-gated skip connections receives the masked image and predicts the full image.
3. **Training signal**: Combined MSE + L1 + SSIM loss for sharp, structurally accurate output. No labels required.
4. **Smart Blending**: At inference, original visible pixels are kept; only masked regions use the model's prediction.

### Model Architecture

```
Encoder                                    Decoder
───────                                    ───────
Conv(3→64)  ──── Attention Gate ────→ Conv(128→64) → Output(64→32→3, Sigmoid)
  ↓ MaxPool                                  ↑ UpConv
Conv(64→128) ─── Attention Gate ────→ Conv(256→128)
  ↓ MaxPool                                  ↑ UpConv
Conv(128→256) ── Attention Gate ────→ Conv(512→256)
  ↓ MaxPool                                  ↑ UpConv
Conv(256→512) ── Attention Gate ────→ Conv(1024→512)
  ↓ MaxPool                                  ↑ UpConv
            Bottleneck: Conv(512→1024→1024) + Dropout
```

Attention gates learn to focus on relevant spatial regions, producing sharper skip connections.

### Consistency Analysis

| Metric | What It Measures |
|--------|-----------------|
| **PSNR (Raw)** | Pixel-level fidelity of model output (higher = better) |
| **SSIM (Raw)** | Perceptual similarity of model output (closer to 1 = better) |
| **PSNR (Blended)** | Quality after smart blending — always higher than raw |
| **SSIM (Blended)** | Structural similarity after blending |
| **Error Heatmap** | Spatial distribution of reconstruction errors |

---

## Configuration

Key constants in `model/utils.py`:

```python
IMG_SIZE   = 128   # Working resolution (128×128)
PATCH_SIZE = 8     # Each mask patch covers 8×8 pixels (256 total patches)
```

Training hyperparameters in `model/train.py` / `colab_train.py`:

```python
EPOCHS     = 100   # Training epochs
BATCH_SIZE = 64    # Batch size (reduce to 32 if GPU OOM)
LR         = 5e-4  # Learning rate with cosine annealing
MASK_RATIO = 0.5   # Default masking ratio
```

---

## Expected Results

After training for 100 epochs with 50% masking on STL-10:

| Metric | Typical Range |
|--------|--------------|
| Train Loss | 0.03 – 0.06 |
| Val Loss | 0.04 – 0.08 |
| PSNR (Raw) | 24 – 30 dB |
| PSNR (Blended) | 28 – 35 dB |
| SSIM (Raw) | 0.80 – 0.93 |
| SSIM (Blended) | 0.90 – 0.97 |

Higher mask ratios make reconstruction harder (lower metrics).

---

## Training Options

```bash
# Full training on STL-10 (recommended)
python model/train.py

# Quick test (10 epochs, CIFAR-10 subset)
python model/train.py --quick

# Custom settings
python model/train.py --epochs 200 --lr 3e-4 --batch_size 32

# Include 100K unlabeled STL-10 images (longer but better)
python model/train.py --unlabeled

# Use CIFAR-10 instead
python model/train.py --dataset cifar10
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA GPU recommended (Colab T4 is free and works great)
- ~2.6 GB disk space for STL-10 dataset

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 32 or 16 |
| Model loads but outputs garbage | Retrain — old 64×64 checkpoints are incompatible |
| Blurry reconstructions | Ensure you trained with the new model (128×128 + attention gates) |
| Colab disconnects | Training auto-saves checkpoints every 25 epochs |

---

## License

This project is provided for educational and research purposes.

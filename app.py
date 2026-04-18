"""
Flask Web Application — Masked Image Reconstruction.

Endpoints:
    GET  /              → Main page (upload + results UI)
    POST /reconstruct   → Process uploaded image, return JSON with paths & metrics
"""

import os
import uuid

import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, url_for
from torchvision import transforms

from model.model import MaskedAutoencoder
from model.utils import (
    IMG_SIZE,
    create_patch_mask,
    apply_mask,
    calculate_psnr,
    calculate_ssim,
    generate_error_heatmap,
    tensor_to_numpy,
    save_tensor_as_image,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── App setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/outputs', exist_ok=True)

# ─── Device & model ─────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

# Image preprocessing (resize to model's expected input)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_model(model_path: str = 'checkpoints/best_model.pth'):
    """Load trained weights (or fall back to an untrained model)."""
    global model
    model = MaskedAutoencoder(in_channels=3).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', '?')
        img_size = checkpoint.get('img_size', '?')
        print(f"[INFO] Model loaded from {model_path}  "
              f"(epoch {epoch}, val_loss {val_loss}, img_size {img_size})")
    else:
        print(f"[WARNING] No checkpoint at {model_path} — using untrained model.")

    model.eval()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_original_image(pil_image, save_path, max_display_size=512):
    """
    Save the original uploaded image at good display resolution.
    Preserves aspect ratio, caps at max_display_size for web display.
    """
    w, h = pil_image.size
    if max(w, h) > max_display_size:
        ratio = max_display_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
    pil_image.save(save_path, format='PNG')


def _save_mask_visualization(image_tensor, mask, save_path):
    """
    Overlay the mask on the original image.
    Visible regions stay normal; masked regions get a translucent red tint.
    """
    img_np = tensor_to_numpy(image_tensor)           # (H, W, 3)
    mask_np = mask.squeeze(0).numpy()                 # (H, W)
    mask_3d = np.stack([mask_np] * 3, axis=2)         # (H, W, 3)

    overlay = img_np.copy()
    # Darken + red tint on masked regions
    overlay[mask_3d == 0] *= 0.3
    red = np.zeros_like(overlay)
    red[:, :, 0] = 0.7
    overlay = np.where(mask_3d == 0, np.clip(overlay + red * 0.5, 0, 1), overlay)

    # Save with PIL for quality
    img_uint8 = (overlay.clip(0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(img_uint8).save(save_path, format='PNG')


def _create_blended_result(original_tensor, reconstructed_tensor, mask):
    """
    Smart blending: keep original pixels in visible regions,
    use model output only for masked regions.
    This produces much sharper results.
    """
    mask_expanded = mask.unsqueeze(0) if mask.dim() == 2 else mask  # (1, H, W)
    # visible regions (mask=1): use original
    # masked regions (mask=0): use reconstruction
    blended = original_tensor * mask_expanded + reconstructed_tensor * (1 - mask_expanded)
    return blended


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    """
    Accept an uploaded image + mask ratio, run reconstruction,
    and return a JSON response with image paths and quality metrics.
    """
    # ── Validate input ───────────────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({'error': 'No image file received.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    mask_ratio = float(request.form.get('mask_ratio', 0.5))
    mask_ratio = max(0.1, min(0.9, mask_ratio))

    try:
        uid = uuid.uuid4().hex[:8]

        # ── Save & load uploaded image ───────────────────────────────────
        upload_path = os.path.join('static', 'uploads', f'{uid}_upload.png')
        file.save(upload_path)
        image = Image.open(upload_path).convert('RGB')

        # Save the original at display resolution (NOT downsized to 128×128)
        def out_path(tag):
            return os.path.join('static', 'outputs', f'{uid}_{tag}.png')

        _save_original_image(image, out_path('original'))

        # Preprocess for model (resize to 128×128)
        image_tensor = preprocess(image)                 # (3, H, W)

        # ── Create mask & apply ──────────────────────────────────────────
        mask, _ = create_patch_mask(mask_ratio=mask_ratio)
        masked_image = apply_mask(image_tensor, mask)

        # ── Run model inference ──────────────────────────────────────────
        with torch.no_grad():
            inp = masked_image.unsqueeze(0).to(device)   # (1, 3, H, W)
            out = model(inp)
            reconstructed = out.squeeze(0).cpu()          # (3, H, W)

        # ── Smart blending: original visible + reconstructed masked ──────
        blended = _create_blended_result(image_tensor, reconstructed, mask)

        # ── Save result images ───────────────────────────────────────────
        save_tensor_as_image(image_tensor,   out_path('model_input'))
        save_tensor_as_image(masked_image,   out_path('masked'))
        save_tensor_as_image(reconstructed,  out_path('reconstructed'))
        save_tensor_as_image(blended,        out_path('blended'))

        # ── Quality metrics ──────────────────────────────────────────────
        orig_np    = tensor_to_numpy(image_tensor)
        recon_np   = tensor_to_numpy(reconstructed)
        blended_np = tensor_to_numpy(blended)

        psnr_recon   = calculate_psnr(orig_np, recon_np)
        ssim_recon   = calculate_ssim(orig_np, recon_np)
        psnr_blended = calculate_psnr(orig_np, blended_np)
        ssim_blended = calculate_ssim(orig_np, blended_np)

        # ── Heatmap & mask visualisation ─────────────────────────────────
        generate_error_heatmap(orig_np, recon_np, out_path('heatmap'))
        _save_mask_visualization(image_tensor, mask, out_path('maskvis'))

        # ── Respond ──────────────────────────────────────────────────────
        def static_url(tag):
            return url_for('static', filename=f'outputs/{uid}_{tag}.png')

        return jsonify({
            'success':            True,
            'original':           static_url('original'),
            'model_input':        static_url('model_input'),
            'masked':             static_url('masked'),
            'reconstructed':      static_url('reconstructed'),
            'blended':            static_url('blended'),
            'heatmap':            static_url('heatmap'),
            'mask_visualization': static_url('maskvis'),
            'psnr':               float(round(psnr_recon, 2)),
            'ssim':               float(round(ssim_recon, 4)),
            'psnr_blended':       float(round(psnr_blended, 2)),
            'ssim_blended':       float(round(ssim_blended, 4)),
            'mask_ratio':         float(mask_ratio),
            'img_size':           IMG_SIZE,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model resolution: {IMG_SIZE}×{IMG_SIZE}")
    print("[INFO] Starting Flask on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)

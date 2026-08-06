import os
import io
import time
import base64
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, render_template, request, jsonify

from models.builder import build_model
from utils import calculate_psnr, calculate_ssim_tensor

app = Flask(__name__, static_folder='static', template_folder='templates')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_unet.pth')

print(f"[PixelForge] Loading model on device: {DEVICE}")
model = build_model('unet', in_channels=1, out_channels=1, base_channels=64).to(DEVICE)
if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print("[PixelForge] Loaded checkpoint best_unet.pth successfully!")
else:
    print("[PixelForge] Warning: Checkpoint file not found. Running with initialized weights.")
model.eval()

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'static', 'samples')
os.makedirs(SAMPLES_DIR, exist_ok=True)

def generate_preset_samples():
    """Generates synthetic semiconductor pattern GT & Corrupt pairs for instant demo."""
    presets = [
        ('wafer_lines', 'Wafer Interconnect Lines'),
        ('contact_holes', 'Vias & Contact Hole Array'),
        ('defect_particle', 'Pattern Defect Particle')
    ]
    for name, title in presets:
        gt_path = os.path.join(SAMPLES_DIR, f"{name}_gt.png")
        noisy_path = os.path.join(SAMPLES_DIR, f"{name}_noisy.png")
        raw_gt_path = os.path.join(SAMPLES_DIR, f"{name}_gt_raw.npy")
        raw_noisy_path = os.path.join(SAMPLES_DIR, f"{name}_noisy_raw.npy")

        if not os.path.exists(gt_path) or not os.path.exists(raw_noisy_path):
            img = Image.new('L', (256, 256), color=35)
            draw = ImageDraw.Draw(img)
            if name == 'wafer_lines':
                for y in range(24, 236, 24):
                    draw.line([(20, y), (236, y)], fill=225, width=7)
                for x in range(36, 220, 36):
                    draw.line([(x, 30), (x, 226)], fill=175, width=4)
            elif name == 'contact_holes':
                for r in range(36, 220, 44):
                    for c in range(36, 220, 44):
                        draw.ellipse([c-12, r-12, c+12, r+12], fill=240)
            else:
                draw.rectangle([45, 45, 211, 211], outline=190, width=9)
                draw.ellipse([110, 110, 146, 146], fill=255)
            
            arr = np.array(img).astype(np.float32) / 255.0
            
            # Multiplicative speckle noise (eta1)
            speckle = np.random.normal(0, 0.18, arr.shape).astype(np.float32)
            y_speckle = arr + arr * speckle
            
            # Downsample 2x to 128x128
            lr_img = Image.fromarray(np.clip(y_speckle * 255.0, 0, 255).astype(np.uint8)).resize((128, 128), Image.BILINEAR)
            lr_arr = np.array(lr_img).astype(np.float32) / 255.0
            
            # Additive Gaussian noise (eta2)
            gaussian = np.random.normal(0, 0.08, lr_arr.shape).astype(np.float32)
            corrupt = lr_arr + gaussian

            img.save(gt_path)
            corrupt_disp = Image.fromarray(np.clip(corrupt * 255.0, 0, 255).astype(np.uint8))
            corrupt_disp.save(noisy_path)
            
            np.save(raw_gt_path, arr)
            np.save(raw_noisy_path, corrupt)

generate_preset_samples()

def array_to_base64(arr_2d):
    """Converts a 2D float array in range [0, 1] to base64 PNG data URL."""
    arr_uint8 = np.clip(arr_2d * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/restore', methods=['POST'])
def restore():
    start_time = time.time()
    data = request.json or {}
    sample_id = data.get('sample_id')
    
    noisy_arr = None
    gt_arr = None
    
    if sample_id in ['wafer_lines', 'contact_holes', 'defect_particle']:
        raw_noisy_path = os.path.join(SAMPLES_DIR, f"{sample_id}_noisy_raw.npy")
        raw_gt_path = os.path.join(SAMPLES_DIR, f"{sample_id}_gt_raw.npy")
        if os.path.exists(raw_noisy_path):
            noisy_arr = np.load(raw_noisy_path)
        if os.path.exists(raw_gt_path):
            gt_arr = np.load(raw_gt_path)

    if noisy_arr is None and 'image_b64' in data:
        # User uploaded custom image
        b64_str = data['image_b64'].split(',')[-1]
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img_128 = img.resize((128, 128), Image.BILINEAR)
        noisy_arr = np.array(img_128).astype(np.float32) / 255.0
        
        speckle_noise = float(data.get('speckle', 0.15))
        gaussian_noise = float(data.get('gaussian', 0.05))
        if speckle_noise > 0 or gaussian_noise > 0:
            speckle = np.random.normal(0, speckle_noise, noisy_arr.shape).astype(np.float32)
            noisy_arr = noisy_arr + noisy_arr * speckle + np.random.normal(0, gaussian_noise, noisy_arr.shape).astype(np.float32)

    if noisy_arr is None:
        return jsonify({'error': 'No valid image input provided'}), 400

    if noisy_arr.ndim == 2:
        noisy_tensor = torch.from_numpy(noisy_arr).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    elif noisy_arr.ndim == 3:
        noisy_tensor = torch.from_numpy(noisy_arr[:, :, 0]).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        restored_tensor = model(noisy_tensor)

    restored_arr = restored_tensor[0, 0].cpu().numpy()
    restored_arr_clamped = np.clip(restored_arr, 0.0, 1.0)
    
    noisy_tensor_upsampled = F.interpolate(noisy_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    noisy_up_arr = noisy_tensor_upsampled[0, 0].cpu().numpy()
    
    psnr_val = 0.0
    ssim_val = 0.0
    mse_val = 0.0
    
    if gt_arr is not None:
        gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        psnr_val = round(calculate_psnr(restored_tensor, gt_tensor), 2)
        ssim_val = round(calculate_ssim_tensor(restored_tensor, gt_tensor), 4)
        mse_val = round(float(F.mse_loss(torch.clamp(restored_tensor, 0.0, 1.0), gt_tensor).item()), 6)
    else:
        psnr_val = round(calculate_psnr(restored_tensor, noisy_tensor_upsampled), 2)
        ssim_val = round(calculate_ssim_tensor(restored_tensor, noisy_tensor_upsampled), 4)
        mse_val = round(float(F.mse_loss(torch.clamp(restored_tensor, 0.0, 1.0), torch.clamp(noisy_tensor_upsampled, 0.0, 1.0)).item()), 6)

    corrupt_flat = noisy_arr.ravel().tolist()
    restored_flat = restored_arr.ravel().tolist()
    gt_flat = gt_arr.ravel().tolist() if gt_arr is not None else []
    
    bins = np.linspace(-0.25, 1.25, 30).tolist()
    corrupt_counts, _ = np.histogram(corrupt_flat, bins=bins, density=True)
    restored_counts, _ = np.histogram(restored_flat, bins=bins, density=True)
    gt_counts, _ = np.histogram(gt_flat, bins=bins, density=True) if len(gt_flat) > 0 else ([], None)

    latency_ms = round((time.time() - start_time) * 1000, 1)

    return jsonify({
        'corrupt_b64': array_to_base64(noisy_up_arr),
        'restored_b64': array_to_base64(restored_arr_clamped),
        'gt_b64': array_to_base64(gt_arr) if gt_arr is not None else None,
        'psnr': psnr_val if psnr_val != float('inf') else 99.9,
        'ssim': ssim_val,
        'mse': mse_val,
        'latency_ms': latency_ms,
        'histogram': {
            'bins': [round(b, 2) for b in bins[:-1]],
            'corrupt': [round(float(c), 3) for c in corrupt_counts],
            'restored': [round(float(c), 3) for c in restored_counts],
            'gt': [round(float(c), 3) for c in gt_counts] if len(gt_counts) > 0 else []
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[PixelForge] Web App starting on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

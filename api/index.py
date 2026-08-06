import os
import io
import time
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import onnxruntime as ort
from skimage.metrics import structural_similarity as ssim_metric

# We don't need static or templates here, Vercel's CDN handles them!
app = Flask(__name__)

# Root directory of the repository (api/ is one level deep)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'unet.onnx')
SAMPLES_DIR = os.path.join(ROOT_DIR, 'public', 'static', 'samples')

ort_session = None
if os.path.exists(MODEL_PATH):
    print(f"[PixelForge] Loading ONNX model from {MODEL_PATH}")
    ort_session = ort.InferenceSession(MODEL_PATH)
else:
    print("[PixelForge] Warning: ONNX model not found! Inference will fail.")

def array_to_base64(arr_2d):
    """Converts a 2D float array in range [0, 1] to base64 PNG data URL."""
    arr_uint8 = np.clip(arr_2d * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

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

    noisy_input = noisy_arr[np.newaxis, np.newaxis, :, :].astype(np.float32)
    
    if ort_session is not None:
        outputs = ort_session.run(None, {'input': noisy_input})
        restored_arr = outputs[0][0, 0]
    else:
        restored_arr = np.zeros((256, 256), dtype=np.float32)

    restored_arr_clamped = np.clip(restored_arr, 0.0, 1.0)
    
    noisy_img_up = Image.fromarray(np.clip(noisy_arr * 255.0, 0, 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR)
    noisy_up_arr = np.array(noisy_img_up).astype(np.float32) / 255.0
    
    psnr_val = 0.0
    ssim_val = 0.0
    mse_val = 0.0
    
    if gt_arr is not None:
        psnr_val = round(calculate_psnr(restored_arr_clamped, gt_arr), 2)
        ssim_val = round(ssim_metric(restored_arr_clamped, gt_arr, data_range=1.0), 4)
        mse_val = round(float(np.mean((restored_arr_clamped - gt_arr)**2)), 6)
    else:
        psnr_val = round(calculate_psnr(restored_arr_clamped, noisy_up_arr), 2)
        ssim_val = round(ssim_metric(restored_arr_clamped, noisy_up_arr, data_range=1.0), 4)
        mse_val = round(float(np.mean((restored_arr_clamped - noisy_up_arr)**2)), 6)

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

# Vercel relies on the 'app' variable being accessible.

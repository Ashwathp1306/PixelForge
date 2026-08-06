import torch
import os
from models.builder import build_model

def main():
    device = torch.device('cpu')
    model_path = os.path.join('checkpoints', 'best_unet.pth')
    onnx_path = os.path.join('checkpoints', 'unet.onnx')
    
    print(f"Loading model from {model_path}...")
    model = build_model('unet', in_channels=1, out_channels=1, base_channels=64).to(device)
    
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Model loaded successfully.")
    else:
        print("Checkpoint not found, exporting with random weights.")
        
    model.eval()
    
    # Dummy input matching the expected input shape: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 1, 128, 128)
    
    print(f"Exporting to ONNX at {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete!")

if __name__ == '__main__':
    main()

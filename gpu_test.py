# Save as gpu_test.py
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
    
    # Quick test
    x = torch.randn(100, 100).cuda()
    print("✅ GPU test successful!")
else:
    print("❌ GPU not detected")

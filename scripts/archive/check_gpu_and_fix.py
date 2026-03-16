#!/usr/bin/env python3
"""
Check GPU availability and provide fix instructions
"""

import torch
import sys

print("="*70)
print("GPU DETECTION CHECK")
print("="*70)

print(f"\n1. PyTorch version: {torch.__version__}")
print(f"2. CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"3. CUDA version: {torch.version.cuda}")
    print(f"4. GPU device: {torch.cuda.get_device_name(0)}")
    print(f"5. GPU count: {torch.cuda.device_count()}")
    print("\n✅ GPU is ready to use!")
else:
    print("\n❌ CUDA NOT AVAILABLE!")
    print("\n🔧 FIX NEEDED:")
    print("\nYou have CPU-only PyTorch installed. To use your RTX 4060:")
    print("\n1. Uninstall current PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("\n2. Install CUDA-enabled PyTorch:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\n   (For CUDA 11.8, use cu118 instead of cu121)")
    print("\n3. Verify installation:")
    print("   python -c \"import torch; print(torch.cuda.is_available())\"")
    print("\n📌 After reinstalling PyTorch with CUDA support:")
    print("   - Your RTX 4060 will be automatically detected")
    print("   - Models will run MUCH faster (10-50x)")
    print("   - No code changes needed!")
    
    print("\n" + "="*70)
    print("SYSTEM INFO")
    print("="*70)
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n✅ NVIDIA Driver installed:")
            print(result.stdout)
        else:
            print("\n❌ nvidia-smi not found - NVIDIA drivers may not be installed")
    except FileNotFoundError:
        print("\n❌ nvidia-smi not found - NVIDIA drivers may not be installed")
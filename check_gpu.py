import os
import sys

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

print("\n--- Environment Variables (PATH) ---")
for p in os.environ["PATH"].split(os.pathsep):
    if "nvidia" in p.lower() or "cuda" in p.lower():
        print(f"  {p}")

print("\n--- Checking CTranslate2 (Faster-Whisper backend) ---")
try:
    import ctranslate2
    print(f"CTranslate2 Version: {ctranslate2.__version__}")
    print(f"CTranslate2 CUDA Available: {ctranslate2.get_cuda_device_count() > 0}")
    print(f"CTranslate2 CUDA Device Count: {ctranslate2.get_cuda_device_count()}")
except ImportError:
    print("CTranslate2 not installed.")
except Exception as e:
    print(f"Error checking CTranslate2: {e}")

print("\n--- Checking torch (if installed) ---")
try:
    import torch
    print(f"Torch Version: {torch.__version__}")
    print(f"Torch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Torch CUDA Version: {torch.version.cuda}")
        print(f"Torch CUDA Device Name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("Torch not installed (not strictly required for faster-whisper but good for debug).")
except Exception as e:
    print(f"Error checking Torch: {e}")

print("\n--- Checking NVIDIA Libraries ---")
try:
    import nvidia.cublas
    import nvidia.cudnn
    print(f"nvidia.cublas file: {nvidia.cublas.__file__}")
    print(f"nvidia.cudnn file: {nvidia.cudnn.__file__}")
except ImportError:
    print("nvidia-cublas or nvidia-cudnn python packages not found.")

print("\n--- Done ---")

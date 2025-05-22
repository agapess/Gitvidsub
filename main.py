import os
import gradio as gr
import torch
import subprocess
import platform
from src.ui import create_ui

def check_gpu_status():
    """Check GPU status and return diagnostic information"""
    status = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available",
        "torch_version": torch.__version__
    }
    
    # Check for NVIDIA drivers on Windows
    if platform.system() == "Windows":
        try:
            nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            status["nvidia_smi"] = "Available" if nvidia_smi.returncode == 0 else "Not available"
            status["nvidia_smi_output"] = nvidia_smi.stdout if nvidia_smi.returncode == 0 else nvidia_smi.stderr
        except:
            status["nvidia_smi"] = "Command not found"
            status["nvidia_smi_output"] = "nvidia-smi command not found"
    
    return status

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Check GPU status
    gpu_status = check_gpu_status()
    print("\n----- GPU Status -----")
    for key, value in gpu_status.items():
        if key != "nvidia_smi_output":  # Skip long output
            print(f"{key}: {value}")
    print("---------------------\n")
    
    if "nvidia_smi_output" in gpu_status and gpu_status["nvidia_smi"] == "Available":
        print("\n----- NVIDIA Driver Information -----")
        print(gpu_status["nvidia_smi_output"])
        print("------------------------------------\n")
    
    # Launch the Gradio interface
    ui = create_ui(gpu_status)
    ui.launch(server_name="127.0.0.1", share=False)
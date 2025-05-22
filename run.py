import os
import sys
import platform
import subprocess
import traceback

def check_environment():
    """Check if the environment is set up correctly"""
    issues = []
    
    # Check Python version
    python_version = platform.python_version_tuple()
    if int(python_version[0]) < 3 or (int(python_version[0]) == 3 and int(python_version[1]) < 8):
        issues.append("Python 3.8 or newer is required.")
    
    # Check virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        issues.append("Not running in a virtual environment. Please activate the venv first.")
    
    # Check for FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except:
        issues.append("FFmpeg is not installed or not in PATH.")
    
    # Check for whisper module conflict
    try:
        import whisper
        if not hasattr(whisper, "load_model"):
            issues.append("Found incorrect whisper module. Please run cleanup.py.")
    except ImportError:
        issues.append("OpenAI Whisper package is not installed.")
    except Exception as e:
        issues.append(f"Error with whisper module: {str(e)}")
    
    # Check for transformers
    try:
        import transformers
    except ImportError:
        issues.append("Transformers package is not installed.")
    
    return issues

def create_models_directory():
    """Create and initialize the models directory"""
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a readme file to explain the directory
    readme_path = os.path.join(models_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("""# Custom Models Directory

This directory is where you can store your custom translation models.

To use a custom model:
1. Save your model in a subdirectory here
2. Select "custom" from the Translation Model dropdown in the app
3. Enter the full path to your model directory
4. Set the appropriate source and target language codes

See CUSTOM_MODELS.md for more details on using custom models.
""")

def main():
    """Main entry point with error handling"""
    print("Checking environment...")
    
    issues = check_environment()
    if issues:
        print("\n⚠️ Environment issues detected:")
        for issue in issues:
            print(f"  • {issue}")
        
        print("\nPlease run setup.bat to fix these issues.")
        return 1
    
    # Create necessary directories
    print("Setting up directories...")
    os.makedirs("output", exist_ok=True)
    create_models_directory()
    
    print("Checking for GPU acceleration...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else "Not available"
        
        print(f"CUDA available: {'✅ Yes' if cuda_available else '❌ No'}")
        print(f"GPU device: {cuda_device}")
    except Exception as e:
        print(f"Error checking GPU: {str(e)}")
        cuda_available = False
    
    try:
        print("\nStarting application...")
        # Import the application
        from src.ui import create_ui
        import gradio as gr
        
        # Create and launch the UI
        ui = create_ui()
        ui.launch(server_name="127.0.0.1", share=False)
        return 0
    
    except Exception as e:
        print(f"\n❌ Error starting the application: {str(e)}")
        print("\nDetailed error information:")
        traceback.print_exc()
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
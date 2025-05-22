# Installation Guide for Video Subtitle Translator

This guide will help you set up the Video Subtitle Translator application on Windows with proper GPU acceleration.

## Prerequisites

1. **Python**: Install Python 3.9 or later from [python.org](https://www.python.org/downloads/)
2. **NVIDIA GPU**: You need an NVIDIA GPU with up-to-date drivers
3. **FFmpeg**: Install FFmpeg with NVENC support

## Step 1: Install NVIDIA Components

1. **Install NVIDIA Drivers**: Download and install the latest drivers for your GPU from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

2. **Install CUDA Toolkit**: Download and install CUDA Toolkit 11.8 from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive)
   - During installation, make sure to select "Custom" installation and ensure that the Visual Studio integration components are selected

3. **Verify CUDA Installation**:
   - Open Command Prompt and run: `nvcc --version`
   - You should see the CUDA version information

## Step 2: Install FFmpeg

1. Download FFmpeg from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/). Choose "ffmpeg-git-full.7z" for full features including NVENC support.
2. Extract the archive to a folder, e.g., `C:\ffmpeg`
3. Add FFmpeg to your PATH:
   - Search for "Environment Variables" in Windows search
   - Edit the "Path" variable
   - Add the path to FFmpeg's bin directory (e.g., `C:\ffmpeg\bin`)
4. Verify FFmpeg installation:
   - Open a new Command Prompt window
   - Run: `ffmpeg -version`
   - Run: `ffmpeg -encoders | findstr nvenc` (should show NVIDIA encoders)

## Step 3: Set Up Python Environment

1. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install PyTorch with CUDA support:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Verify PyTorch CUDA support:
   ```python
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```
   - This should print `CUDA available: True`

4. Install other requirements:
   ```
   pip install -r requirements.txt
   ```

## Step 4: Running the Application

1. Run the application:
   ```
   python main.py
   ```

2. Access the WebUI at http://127.0.0.1:7860

## Troubleshooting

If you encounter GPU acceleration issues:

1. **CUDA not found**:
   - Make sure NVIDIA drivers are installed
   - Check that CUDA Toolkit is installed and in your PATH
   - Verify PyTorch was installed with CUDA support

2. **FFmpeg NVENC issues**:
   - Check if your FFmpeg build includes NVENC support: `ffmpeg -encoders | findstr nvenc`
   - Ensure your GPU supports NVENC

3. **Application uses CPU even with CUDA available**:
   - Check if PyTorch recognizes your GPU: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
   - Look at the application's startup logs for GPU status

4. **Subtitle burning fails**:
   - Try converting SRT to ASS format manually:
     ```
     ffmpeg -i subtitles.srt subtitles.ass
     ```
   - Use the ASS file in the FFmpeg command
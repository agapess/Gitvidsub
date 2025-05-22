@echo off
echo Setting up Video Subtitle Translator...
echo.

echo Checking Python installation...
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python first.
    pause
    exit /b
)

echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment. Please make sure venv module is available.
    pause
    exit /b
)

echo Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b
)

echo Checking for NVIDIA GPU...
nvidia-smi > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected. Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo NVIDIA GPU not detected. Installing PyTorch for CPU...
    pip install torch torchvision torchaudio
)

echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo Checking FFmpeg installation...
ffmpeg -version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FFmpeg is not installed or not in PATH. Please install FFmpeg.
    echo See INSTALLATION.md for details.
)

echo.
echo Setup complete! Run the application with:
echo python main.py
echo.
pause
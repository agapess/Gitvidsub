@echo off
echo ========================================
echo Video Subtitle Translator Setup
echo ========================================
echo.

:: Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.9 or later from python.org.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Fix whisper module conflict
echo Checking for module conflicts...
python cleanup.py
if %ERRORLEVEL% NEQ 0 (
    echo Failed to check for module conflicts.
    pause
    exit /b 1
)

:: Install or update pip
echo Updating pip...
python -m pip install --upgrade pip

:: Install PyTorch with CUDA
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected. Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo NVIDIA GPU not detected or drivers not installed.
    echo Installing PyTorch for CPU...
    pip install torch torchvision torchaudio
)

:: Install OpenAI Whisper
echo Installing OpenAI Whisper...
pip install openai-whisper

:: Install other requirements
echo Installing other dependencies...
pip install gradio>=4.0.0 transformers>=4.30.0 yt-dlp>=2023.3.4 ffmpeg-python>=0.2.0 pysrt>=1.1.2 sacremoses>=0.0.53 sentencepiece>=0.1.99

:: Check FFmpeg
echo Checking for FFmpeg...
ffmpeg -version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: FFmpeg is not installed or not in PATH.
    echo You need to install FFmpeg for this application to work.
    echo Download from: https://www.gyan.dev/ffmpeg/builds/
    echo Download the "ffmpeg-git-full.7z" package, extract it,
    echo and add the bin folder to your PATH.
    echo.
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To run the application:
echo   venv\Scripts\activate
echo   python run.py
echo.
pause
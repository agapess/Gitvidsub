import os
import torch
from datetime import timedelta

# Try different import strategies for whisper
try:
    import openai.whisper as whisper
except ImportError:
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "Could not import whisper module. Please install it using: pip install openai-whisper"
        )

# Import pysrt for subtitle handling
try:
    import pysrt
except ImportError:
    raise ImportError("Could not import pysrt module. Please install it using: pip install pysrt")

def transcribe_audio(video_path, model_name="large", output_dir="output", device=None):
    """
    Generate subtitles from video/audio file using Whisper
    
    Args:
        video_path (str): Path to the video/audio file
        model_name (str): Whisper model name (tiny, base, small, medium, large)
        output_dir (str): Directory to save the subtitle file
        device (str): Device to use for inference (cuda or cpu)
        
    Returns:
        str: Path to the generated subtitle file
    """
    # Determine the device to use
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # If CPU is being used, recommend a smaller model
    if device == "cpu" and model_name == "large":
        print("Warning: Running the large model on CPU will be very slow. Consider using a smaller model like 'base' or 'small'.")
        # If on CPU, default to a smaller model unless explicitly specified
        if not os.environ.get("FORCE_LARGE_MODEL"):
            print("Automatically switching to 'small' model for CPU. Set FORCE_LARGE_MODEL=1 to override.")
            model_name = "small"
    
    print(f"Loading Whisper {model_name} model on {device}...")
    
    try:
        # Ensure the model is loaded correctly
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        # Handle errors related to model loading
        print(f"Error loading whisper model: {str(e)}")
        print("\nIf this is related to module conflicts, ensure you're using openai-whisper package.")
        raise
    
    print(f"Transcribing {video_path}...")
    # Use fp16=False on CPU to avoid errors
    use_fp16 = torch.cuda.is_available() if device == "cuda" else False
    result = model.transcribe(video_path, fp16=use_fp16)
    
    # Create SRT file
    srt_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + '.srt')
    
    # Create subtitle entries
    subs = pysrt.SubRipFile()
    for i, segment in enumerate(result["segments"]):
        start_time = timedelta(seconds=segment["start"])
        end_time = timedelta(seconds=segment["end"])
        
        # Create a new subtitle entry
        sub = pysrt.SubRipItem()
        sub.index = i + 1
        sub.start.seconds = start_time.seconds
        sub.start.minutes = int(start_time.seconds / 60)
        sub.start.hours = int(start_time.seconds / 3600)
        sub.start.milliseconds = int((start_time.microseconds / 1000) % 1000)
        
        sub.end.seconds = end_time.seconds
        sub.end.minutes = int(end_time.seconds / 60)
        sub.end.hours = int(end_time.seconds / 3600)
        sub.end.milliseconds = int((end_time.microseconds / 1000) % 1000)
        
        sub.text = segment["text"]
        subs.append(sub)
    
    # Save the SRT file
    subs.save(srt_path, encoding='utf-8')
    
    return srt_path
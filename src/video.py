import os
import subprocess
import platform
import shutil

def check_ffmpeg_encoders():
    """Check if NVENC is available in the FFmpeg build"""
    try:
        process = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in process.stdout
    except:
        return False

def burn_subtitles(video_path, subtitle_path, output_dir="output", style="default"):
    """
    Burn subtitles into the video using FFmpeg
    
    Args:
        video_path (str): Path to the video file
        subtitle_path (str): Path to the subtitle file
        output_dir (str): Directory to save the output video
        style (str): Subtitle style (default, fade, slide)
        
    Returns:
        str: Path to the output video with burned subtitles
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output video path
    output_video = os.path.join(
        output_dir, 
        os.path.splitext(os.path.basename(video_path))[0] + '_subbed.mp4'
    )
    
    # Define subtitle style options
    style_options = {
        "default": "",
        "fade": ":force_style='Fontname=Arial,FontSize=24,PrimaryColour=&HFFFFFF,BackColour=&H80000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=30,Fade(1000,1000)'",
        "slide": ":force_style='Fontname=Arial,FontSize=24,PrimaryColour=&HFFFFFF,BackColour=&H80000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=30,Alignment=2'",
        "outline": ":force_style='Fontname=Arial,FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=30'"
    }
    
    style_option = style_options.get(style, "")
    
    # Check if FFmpeg exists
    if not shutil.which("ffmpeg"):
        raise Exception("FFmpeg is not installed. Please install FFmpeg and make sure it's in your PATH.")
    
    # Check if NVENC is available
    nvenc_available = check_ffmpeg_encoders()
    
    # Fix paths for Windows (escape backslashes and quotes)
    if platform.system() == "Windows":
        # Normalize paths for Windows
        subtitle_path = os.path.normpath(subtitle_path).replace('\\', '\\\\')
        output_video = os.path.normpath(output_video)
        
        # Create subtitle filter with properly escaped path
        subtitle_filter = f"subtitles={subtitle_path}{style_option}"
    else:
        # For non-Windows systems, quote the path
        subtitle_filter = f"subtitles='{subtitle_path}'{style_option}"
    
    # Base FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", subtitle_filter
    ]
    
    # Add encoder options based on availability
    if nvenc_available:
        print("Using NVIDIA hardware acceleration (NVENC)")
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "slow", "-crf", "18"])
    else:
        print("NVIDIA hardware acceleration not available. Using CPU encoding.")
        cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])
    
    # Add audio codec and output path
    cmd.extend(["-c:a", "copy", output_video])
    
    print(f"Running FFmpeg command: {' '.join(cmd)}")
    
    try:
        # Run FFmpeg command and capture output
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output for debugging
        if process.returncode != 0:
            print(f"FFmpeg error: {process.stderr}")
            raise Exception(f"FFmpeg failed with error: {process.stderr}")
            
        return output_video
    except Exception as e:
        print(f"Error burning subtitles: {str(e)}")
        
        # Try alternate method with simpler parameters
        try:
            # Prepare alternate command based on available encoders
            cmd_alt = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"subtitles={os.path.basename(subtitle_path)}"
            ]
            
            if nvenc_available:
                cmd_alt.extend(["-c:v", "h264_nvenc"])
            else:
                cmd_alt.extend(["-c:v", "libx264", "-preset", "fast"])
                
            cmd_alt.extend(["-c:a", "copy", output_video])
            
            # Change working directory to where the subtitle file is located
            original_dir = os.getcwd()
            os.chdir(os.path.dirname(subtitle_path))
            
            print(f"Trying alternate FFmpeg command: {' '.join(cmd_alt)}")
            subprocess.run(cmd_alt, check=True)
            
            # Change back to original directory
            os.chdir(original_dir)
            
            return output_video
        except Exception as e2:
            print(f"Alternate method failed: {str(e2)}")
            
            # Last resort: try with absolute path and minimal options
            try:
                cmd_final = [
                    "ffmpeg", "-y",
                    "-i", os.path.abspath(video_path),
                    "-vf", f"subtitles={os.path.abspath(subtitle_path)}",
                    "-c:v", "libx264",
                    "-c:a", "copy",
                    os.path.abspath(output_video)
                ]
                
                print(f"Trying final FFmpeg command: {' '.join(cmd_final)}")
                subprocess.run(cmd_final, check=True)
                
                return output_video
            except Exception as e3:
                print(f"Final method failed: {str(e3)}")
                raise e
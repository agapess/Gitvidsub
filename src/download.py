import os
import yt_dlp

def download_video(url, output_dir="output"):
    """
    Download videos from YouTube, Instagram, or other supported platforms
    
    Args:
        url (str): URL of the video to download
        output_dir (str): Directory to save the downloaded video
        
    Returns:
        tuple: (str: Path to the downloaded video file, str: Status message)
    """
    # Check if URL is empty
    if not url or not url.strip():
        return None, "Error: No URL provided."
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': True,
            'noplaylist': True,
        }
        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if 'entries' in info:
                # Playlist
                info = info['entries'][0]
            
            # Get the downloaded file path
            downloaded_file = os.path.join(output_dir, f"{info['title']}.{info['ext']}")
            status_message = f"Successfully downloaded: {info['title']}"
            
            # Verify file exists
            if os.path.exists(downloaded_file):
                return downloaded_file, status_message
            else:
                return None, f"Error: File was not saved at expected location: {downloaded_file}"
            
    except Exception as e:
        error_message = f"Error downloading video: {str(e)}"
        print(error_message)
        return None, error_message
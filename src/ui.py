import os
import gradio as gr
import torch
import platform
from src.download import download_video
from src.transcribe import transcribe_audio
from src.translate import translate_subtitles, TRANSLATION_MODELS
from src.video import burn_subtitles

def safe_download_video(url, output_dir):
    """
    Wrapper for download_video that ensures proper return values
    
    Args:
        url (str): URL of the video to download
        output_dir (str): Directory to save the downloaded video
        
    Returns:
        tuple: (downloaded_path, status_message)
    """
    try:
        result = download_video(url, output_dir)
        
        # Check if the function returned the expected tuple
        if isinstance(result, tuple) and len(result) == 2:
            return result
        
        # If not a proper tuple, create a proper response
        if result is None:
            return None, "Error: Download function returned None"
        
        # If only one value was returned (old version of function)
        if not isinstance(result, tuple):
            return result, f"Downloaded: {result}"
            
        # Otherwise return None with error
        return None, "Error: Invalid response from download function"
        
    except Exception as e:
        print(f"Error in download function: {str(e)}")
        return None, f"Error downloading: {str(e)}"

def create_ui(gpu_status=None):
    """
    Create the Gradio UI for the video translation app
    
    Args:
        gpu_status (dict): Dictionary containing GPU status information
        
    Returns:
        gr.Blocks: Gradio interface
    """
    # If gpu_status is None, create a default one
    if gpu_status is None:
        gpu_status = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available"
        }
    
    # Check if CUDA is available
    cuda_available = gpu_status.get("cuda_available", False)
    cuda_device = gpu_status.get("cuda_device_name", "Not available")
    
    # Default output directory
    default_output_dir = os.path.join(os.getcwd(), "output")
    
    with gr.Blocks(title="Video Subtitle Translator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Video Subtitle Translator")
        gr.Markdown(f"**GPU Acceleration: {'✅ Enabled' if cuda_available else '❌ Disabled'} ({cuda_device})**")
        
        if not cuda_available:
            gr.Markdown("""
            > ⚠️ **Warning:** GPU acceleration is not available. The application will still work but will be significantly slower.
            > 
            > **Possible solutions:**
            > 1. Make sure your NVIDIA drivers are installed
            > 2. Install CUDA Toolkit (11.8 or compatible with your PyTorch version)
            > 3. Reinstall PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
            """)
        
        with gr.Tabs():
            with gr.TabItem("File Upload"):
                video_file = gr.File(label="Upload Video/Audio File")
            
            with gr.TabItem("YouTube/Instagram URL"):
                video_url = gr.Textbox(label="Enter Video URL (YouTube, Instagram, etc.)")
                download_btn = gr.Button("Download Video")
                download_status = gr.Textbox(label="Download Status", interactive=False)
                downloaded_path = gr.Textbox(label="Downloaded File Path", visible=False)
        
        with gr.Accordion("Advanced Options", open=False):
            whisper_model = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large"], 
                value="small" if not cuda_available else "medium",
                label="Whisper Model Size"
            )
            
            translation_model = gr.Dropdown(
                choices=list(TRANSLATION_MODELS.keys()), 
                value="mt5-parsinlu-large", 
                label="Translation Model"
            )
            
            # Custom model section that shows only when "custom" is selected
            with gr.Group(visible=False) as custom_model_group:
                custom_model_path = gr.Textbox(
                    label="Custom Model Path/HF Model ID",
                    placeholder="Path to local model directory or Hugging Face model ID",
                    value=""
                )
                
                with gr.Row():
                    custom_source_lang = gr.Textbox(
                        label="Source Language Code", 
                        value="en",
                        placeholder="e.g., en, en_XX, eng_Latn"
                    )
                    
                    custom_target_lang = gr.Textbox(
                        label="Target Language Code", 
                        value="fa",
                        placeholder="e.g., fa, fa_IR, fas_Arab"
                    )
            
            # Update visibility of custom model options based on selection
            def update_custom_model_visibility(model_choice):
                return gr.Group.update(visible=(model_choice == "custom"))
            
            translation_model.change(
                fn=update_custom_model_visibility,
                inputs=translation_model,
                outputs=custom_model_group
            )
            
            subtitle_style = gr.Dropdown(
                choices=["default", "fade", "slide", "outline"], 
                value="default", 
                label="Subtitle Animation Style"
            )
            
            output_directory = gr.Textbox(
                label="Output Directory", 
                value=default_output_dir, 
                placeholder="Path to save the output files"
            )
        
        process_btn = gr.Button("Process Video", variant="primary")
        
        with gr.Accordion("Processing Status", open=True):
            status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Accordion("Output", open=True):
            output_video = gr.Video(label="Processed Video with Persian Subtitles")
            output_path = gr.Textbox(label="Output File Path", interactive=False)
        
        # Download video from URL
        download_btn.click(
            fn=safe_download_video,
            inputs=[video_url, output_directory],
            outputs=[downloaded_path, download_status]
        )
        
        def process_video(video_file, video_url, downloaded_path, whisper_model, translation_model, 
                          custom_model_path, custom_source_lang, custom_target_lang, subtitle_style, output_dir):
            """Process the video: transcribe, translate, and burn subtitles"""
            try:
                # Make sure output_dir is an absolute path
                if not os.path.isabs(output_dir):
                    output_dir = os.path.abspath(output_dir)
                    
                # Check if output directory exists, create it if not
                os.makedirs(output_dir, exist_ok=True)
                
                # Determine the video path from either file upload or URL download
                if video_file:
                    video_path = video_file
                    yield f"Using uploaded video file: {os.path.basename(video_path)}", None, None
                elif downloaded_path and os.path.exists(downloaded_path):
                    video_path = downloaded_path
                    yield f"Using downloaded video from URL: {video_url}", None, None
                else:
                    yield "Error: Please upload a valid video file or provide a valid URL", None, None
                    return
                
                # Transcribe the video to generate subtitles
                yield "Transcribing video with Whisper...", None, None
                srt_path = transcribe_audio(video_path, whisper_model, output_dir, device="cuda" if torch.cuda.is_available() else "cpu")
                yield f"Transcription completed. Subtitle file saved to {srt_path}", None, None
                
                # Translate the subtitles
                target_lang_display = custom_target_lang if translation_model == "custom" else TRANSLATION_MODELS[translation_model]["target_lang"]
                yield f"Translating subtitles to {target_lang_display} using {translation_model}...", None, None
                
                translated_srt_path = translate_subtitles(
                    srt_path, 
                    translation_model, 
                    output_dir,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    custom_model_path=custom_model_path if translation_model == "custom" else None,
                    custom_source_lang=custom_source_lang,
                    custom_target_lang=custom_target_lang
                )
                
                yield f"Translation completed. Translated subtitle file saved to {translated_srt_path}", None, None
                
                # Burn the subtitles into the video
                yield "Burning subtitles into the video...", None, None
                output_video_path = burn_subtitles(video_path, translated_srt_path, output_dir, subtitle_style)
                
                # Verify the file exists
                if os.path.exists(output_video_path):
                    yield f"Processing completed! Video with subtitles saved to {output_video_path}", output_video_path, output_video_path
                else:
                    yield f"Error: Output file was not created at {output_video_path}. Please check logs.", None, None
                    
            except Exception as e:
                yield f"Error processing video: {str(e)}", None, None
        
        # Process video button
        process_btn.click(
            fn=process_video,
            inputs=[
                video_file, 
                video_url, 
                downloaded_path, 
                whisper_model,
                translation_model,
                custom_model_path,
                custom_source_lang,
                custom_target_lang,
                subtitle_style,
                output_directory
            ],
            outputs=[status, output_path, output_video]
        )
    
    return interface

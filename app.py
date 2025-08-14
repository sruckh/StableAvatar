import gradio as gr
import os
import subprocess
import librosa
import shutil
from pathlib import Path

# Initialize the application
def init_app():
    """Initialize the application directories and environment"""
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Set Python stream buffer to 1
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    print("Application initialized successfully!")

init_app()

def run_inference(prompt, reference_image, audio_file, width=512, height=512, sample_steps=50):
    """
    Run the StableAvatar inference script with the given parameters
    """
    # Save uploaded files to temporary locations
    ref_img_path = "temp/reference.png"
    audio_path = "temp/audio.wav"
    output_dir = "output/inference_result"
    
    # Save the uploaded files
    reference_image.save(ref_img_path)
    audio_file.save(audio_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the inference command with proper arguments
    cmd = [
        "python", "inference.py",
        "--config_path=deepspeed_config/wan2.1/wan_civitai.yaml",
        f"--pretrained_model_name_or_path=./checkpoints/Wan2.1-Fun-V1.1-1.3B-InP",
        f"--transformer_path=./checkpoints/StableAvatar-1.3B/transformer3d-square.pt",
        f"--pretrained_wav2vec_path=./checkpoints/wav2vec2-base-960h",
        f"--validation_reference_path={ref_img_path}",
        f"--validation_driven_audio_path={audio_path}",
        f"--output_dir={output_dir}",
        f"--validation_prompts={prompt}",
        "--seed=42",
        "--ulysses_degree=1",
        "--ring_degree=1",
        "--motion_frame=25",
        f"--sample_steps={sample_steps}",
        f"--width={width}",
        f"--height={height}",
        "--overlap_window_length=5",
        "--clip_sample_n_frames=81",
        "--GPU_memory_mode=model_full_load",
        "--sample_text_guide_scale=5.0",
        "--sample_audio_guide_scale=3.0"
    ]
    
    try:
        print(f"Running inference command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            cwd=".",
            capture_output=True, 
            text=True, 
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            # Find the generated video
            video_path = os.path.join(output_dir, "video_without_audio.mp4")
            if os.path.exists(video_path):
                return video_path, "Inference completed successfully!"
            else:
                # Try to find any mp4 file in the output directory
                mp4_files = list(Path(output_dir).glob("*.mp4"))
                if mp4_files:
                    return str(mp4_files[0]), "Inference completed successfully!"
                else:
                    return None, f"Video file not found. Command output: {result.stdout[:500]}"
        else:
            return None, f"Error running inference: {result.stderr[:500]}"
            
    except subprocess.TimeoutExpired:
        return None, "Inference timed out after 1 hour"
    except Exception as e:
        return None, f"Error running inference: {str(e)}"

def extract_audio(video_file):
    """
    Extract audio from video file using audio_extractor.py
    """
    # Save uploaded video to temporary location
    video_path = "temp/input_video.mp4"
    audio_output_path = "temp/extracted_audio.wav"
    
    with open(video_path, "wb") as f:
        with open(video_file.name, "rb") as vf:
            f.write(vf.read())
    
    # Run audio extraction command
    cmd = [
        "python", "audio_extractor.py",
        f"--video_path={video_path}",
        f"--saved_audio_path={audio_output_path}"
    ]
    
    try:
        print(f"Running audio extraction command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(audio_output_path):
            return audio_output_path, "Audio extracted successfully!"
        else:
            return None, f"Error extracting audio: {result.stderr[:500]}"
            
    except subprocess.TimeoutExpired:
        return None, "Audio extraction timed out"
    except Exception as e:
        return None, f"Error extracting audio: {str(e)}"

def separate_vocals(audio_file):
    """
    Separate vocals from audio file using vocal_seperator.py
    """
    # Save uploaded audio to temporary location
    audio_path = "temp/input_audio.wav"
    vocal_output_path = "temp/vocal_output.wav"
    
    with open(audio_path, "wb") as f:
        with open(audio_file.name, "rb") as af:
            f.write(af.read())
    
    # Run vocal separation command
    model_path = "./checkpoints/Kim_Vocal_2.onnx"
    
    cmd = [
        "python", "vocal_seperator.py",
        f"--audio_separator_model_file={model_path}",
        f"--audio_file_path={audio_path}",
        f"--saved_vocal_path={vocal_output_path}"
    ]
    
    try:
        print(f"Running vocal separation command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(vocal_output_path):
            return vocal_output_path, "Vocal separation completed successfully!"
        else:
            return None, f"Error separating vocals: {result.stderr[:500]}"
            
    except subprocess.TimeoutExpired:
        return None, "Vocal separation timed out"
    except Exception as e:
        return None, f"Error separating vocals: {str(e)}"

def merge_audio_video(video_file, audio_file):
    """
    Merge audio with video using ffmpeg
    """
    # Save uploaded files to temporary locations
    video_path = "temp/input_video.mp4"
    audio_path = "temp/input_audio.wav"
    output_path = "output/final_output.mp4"
    
    # Save files
    with open(video_path, "wb") as f:
        with open(video_file.name, "rb") as vf:
            f.write(vf.read())
        
    with open(audio_path, "wb") as f:
        with open(audio_file.name, "rb") as af:
            f.write(af.read())
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run ffmpeg command
    cmd = [
        "ffmpeg", "-i", video_path, "-i", audio_path,
        "-c:v", "libx265", "-pix_fmt", "yuv420p",
        "-vf", "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,scale=out_color_matrix=bt709",
        "-maxrate:v", "3500k", "-bufsize", "6000k",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-threads", "0", "-crf", "22", "-bf", "2", "-g", "15",
        "-movflags", "+faststart", "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0", output_path,
        "-y"  # Overwrite output file
    ]
    
    try:
        print(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path, "Audio and video merged successfully!"
        else:
            return None, f"Error merging audio and video: {result.stderr[:500]}"
            
    except subprocess.TimeoutExpired:
        return None, "Audio/video merge timed out"
    except Exception as e:
        return None, f"Error merging audio and video: {str(e)}"

# Define the Gradio interface
with gr.Blocks(title="StableAvatar Interface") as demo:
    gr.Markdown("# StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation")
    gr.Markdown("Generate avatar videos from reference images and audio files")
    
    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here",
                    value="A person is speaking in front of a blurred background. The scene gives the impression they are singing with conviction and purpose."
                )
                gr.Markdown("[Description of first frame]-[Description of human behavior]-[Description of background (optional)]")
                
                reference_image = gr.Image(type="pil", label="Reference Image")
                audio_file = gr.Audio(type="filepath", label="Audio File")
                
                with gr.Row():
                    width = gr.Number(value=512, label="Width")
                    height = gr.Number(value=512, label="Height")
                    sample_steps = gr.Number(value=50, label="Sample Steps")
                
                run_button = gr.Button("Generate Video")
                
            with gr.Column():
                output_video = gr.Video(label="Generated Video")
                status = gr.Textbox(label="Status", interactive=False)
                # download_button = gr.Button("Download Final Video")
        
        run_button.click(
            fn=run_inference,
            inputs=[prompt, reference_image, audio_file, width, height, sample_steps],
            outputs=[output_video, status]
        )
    
    with gr.Tab("Audio Extraction"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input Video")
                extract_button = gr.Button("Extract Audio")
                
            with gr.Column():
                extracted_audio = gr.Audio(label="Extracted Audio")
                extraction_status = gr.Textbox(label="Status", interactive=False)
        
        extract_button.click(
            fn=extract_audio,
            inputs=[video_input],
            outputs=[extracted_audio, extraction_status]
        )
    
    with gr.Tab("Vocal Separation"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Input Audio")
                separate_button = gr.Button("Separate Vocals")
                
            with gr.Column():
                vocal_output = gr.Audio(label="Vocal Output")
                separation_status = gr.Textbox(label="Status", interactive=False)
        
        separate_button.click(
            fn=separate_vocals,
            inputs=[audio_input],
            outputs=[vocal_output, separation_status]
        )
    
    with gr.Tab("Audio/Video Merge"):
        with gr.Row():
            with gr.Column():
                merge_video_input = gr.Video(label="Input Video")
                merge_audio_input = gr.Audio(type="filepath", label="Input Audio")
                merge_button = gr.Button("Merge Audio and Video")
                
            with gr.Column():
                merged_output = gr.Video(label="Merged Output")
                merge_status = gr.Textbox(label="Status", interactive=False)
        
        merge_button.click(
            fn=merge_audio_video,
            inputs=[merge_video_input, merge_audio_input],
            outputs=[merged_output, merge_status]
        )

# Launch the app when run directly
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
import torch
import psutil
import argparse
import gradio as gr
import os
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from omegaconf import OmegaConf
from wan.models.cache_utils import get_teacache_coefficients
from wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
from wan.models.wan_text_encoder import WanT5EncoderModel
from wan.models.wan_vae import AutoencoderKLWan
from wan.models.wan_image_encoder import CLIPModel
from wan.pipeline.wan_inference_long_pipeline import WanI2VTalkingInferenceLongPipeline
from wan.utils.fp8_optimization import replace_parameters_by_name, convert_weight_dtype_wrapper, convert_model_weight_to_float8
from wan.utils.utils import get_image_to_video_latent, save_videos_grid
import numpy as np
import librosa
import datetime
import random
import math
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--server_name", type=str, default="0.0.0.0", help="IP address, change to 0.0.0.0 for LAN access")
parser.add_argument("--server_port", type=int, default=7860, help="Port to use")
parser.add_argument("--share", type=bool, default=True, help="Enable gradio share")
parser.add_argument("--mcp_server", action="store_true", help="Enable mcp server")
args = parser.parse_args()


if torch.cuda.is_available():
    device = "cuda"
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


model_path = "checkpoints"
pretrained_model_name_or_path = f"{model_path}/Wan2.1-Fun-V1.1-1.3B-InP"
pretrained_wav2vec_path = f"{model_path}/wav2vec2-base-960h"
transformer_path = f"{model_path}/StableAvatar-1.3B/transformer3d-square.pt"
config = OmegaConf.load("deepspeed_config/wan2.1/wan_civitai.yaml")
sampler_name = "Flow"
clip_sample_n_frames = 81
tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')), )
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=dtype,
)
text_encoder = text_encoder.eval()
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
)
wav2vec_processor = Wav2Vec2Processor.from_pretrained(pretrained_wav2vec_path)
wav2vec = Wav2Vec2Model.from_pretrained(pretrained_wav2vec_path).to("cpu")
clip_image_encoder = CLIPModel.from_pretrained(os.path.join(pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')), )
clip_image_encoder = clip_image_encoder.eval()
transformer3d = WanTransformer3DFantasyModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=False,
    torch_dtype=dtype,
)
if transformer_path is not None:
    state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = transformer3d.load_state_dict(state_dict, strict=False)
Choosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler(
    **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)
pipeline = WanI2VTalkingInferenceLongPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    transformer=transformer3d,
    clip_image_encoder=clip_image_encoder,
    scheduler=scheduler,
    wav2vec_processor=wav2vec_processor,
    wav2vec=wav2vec,
)


def merge_audio_video(video_path, audio_path):
    """
    Merge audio with video using the high-quality ffmpeg command from GOALS.md.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/final_output_{timestamp}.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    cmd = [
        "ffmpeg", "-i", video_path, "-i", audio_path,
        "-c:v", "libx265", "-pix_fmt", "yuv420p",
        "-vf", "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,scale=out_color_matrix=bt709",
        "-maxrate:v", "3500k", "-bufsize", "6000k",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-threads", "0", "-crf", "22", "-bf", "2", "-g", "15",
        "-movflags", "+faststart", "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0", output_path,
        "-y"
    ]
    
    try:
        print(f"Running ffmpeg merge command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path, "Audio and video merged successfully!"
        else:
            return None, f"Error merging audio and video: {result.stderr}"
            
    except Exception as e:
        return None, f"Error merging audio and video: {str(e)}"

def generate(
    GPU_memory_mode,
    teacache_threshold,
    num_skip_start_steps,
    image_path,
    audio_path,
    prompt,
    negative_prompt,
    merge_audio,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    text_guide_scale,
    audio_guide_scale,
    motion_frame,
    fps,
    overlap_window_length,
    seed_param,
):
    global pipeline, transformer3d
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed_param<0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param

    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer3d, ["modulation", ], device=device)
        transformer3d.freqs = transformer3d.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer3d, exclude_module_name=["modulation", ])
        convert_weight_dtype_wrapper(transformer3d, dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    if teacache_threshold > 0:
        coefficients = get_teacache_coefficients(pretrained_model_name_or_path)
        pipeline.transformer.enable_teacache(
            coefficients,
            num_inference_steps,
            teacache_threshold,
            num_skip_start_steps=num_skip_start_steps,
            #offload=args.teacache_offload
        )

    with torch.no_grad():
        video_length = int((clip_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if clip_sample_n_frames != 1 else 1
        input_video, input_video_mask, clip_image = get_image_to_video_latent(image_path, None, video_length=video_length, sample_size=[height, width])
        sr = 16000
        vocal_input, sample_rate = librosa.load(audio_path, sr=sr)
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            num_inference_steps=num_inference_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
            vocal_input_values=vocal_input,
            motion_frame=motion_frame,
            fps=fps,
            sr=sr,
            cond_file_path=image_path,
            overlap_window_length=overlap_window_length,
            seed=seed,
            overlapping_weight_scheme="uniform",
        ).videos
        os.makedirs("outputs", exist_ok=True)
        video_path_no_audio = os.path.join("outputs", f"video_no_audio_{timestamp}.mp4")
        save_videos_grid(sample, video_path_no_audio, fps=fps)

        if merge_audio:
            final_video_path, merge_status = merge_audio_video(video_path_no_audio, audio_path)
            if final_video_path:
                return final_video_path, seed, f"Successfully generated and merged video! {merge_status}"
            else:
                return video_path_no_audio, seed, f"Generated video, but merge failed: {merge_status}"
        else:
            return video_path_no_audio, seed, "Successfully generated video (audio not merged)."


def exchange_width_height(width, height):
    return height, width, "✅ Width and height exchanged"


def adjust_width_height(image):
    image = load_image(image)
    width, height = image.size
    original_area = width * height
    default_area = 512*512
    ratio = math.sqrt(original_area / default_area)
    width = width / ratio // 16 * 16
    height = height / ratio // 16 * 16
    return int(width), int(height), "✅ Adjusted width and height based on image"


with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">StableAvatar</h2>
            </div>
            """)
    with gr.Accordion("Model Settings", open=False):
        with gr.Row():
            GPU_memory_mode = gr.Dropdown(
                label = "GPU Memory Mode",
                info = "Normal uses 25G VRAM, model_cpu_offload uses 13G VRAM",
                choices = ["Normal", "model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                value = "model_cpu_offload"
            )
            teacache_threshold = gr.Slider(label="teacache threshold", info = "Recommended: 0.1, 0 disables teacache acceleration", minimum=0, maximum=1, step=0.01, value=0)
            num_skip_start_steps = gr.Slider(label="Skip Start Steps", info = "Recommended: 5", minimum=0, maximum=100, step=1, value=5)
    with gr.TabItem("StableAvatar"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_path = gr.Image(label="Upload Image", type="filepath", height=280)
                    audio_path = gr.Audio(label="Upload Audio", type="filepath")
                prompt = gr.Textbox(label="Prompt", value="")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="Low quality, bad quality, blur, blurry, (deformed iris, deformed pupils), (worst quality, low quality, normal quality), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck")
                generate_button = gr.Button("🎬 Generate", variant='primary')
                merge_audio_checkbox = gr.Checkbox(label="Merge audio into final video", value=True)
                with gr.Accordion("Parameter Settings", open=True):
                    with gr.Row():
                        width = gr.Slider(label="Width", minimum=256, maximum=2048, step=16, value=512)
                        height = gr.Slider(label="Height", minimum=256, maximum=2048, step=16, value=512)
                    with gr.Row():
                        exchange_button = gr.Button("🔄 Swap Width/Height")
                        adjust_button = gr.Button("Adjust Width/Height based on Image")
                    with gr.Row():
                        guidance_scale = gr.Slider(label="guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=6.0)
                        num_inference_steps = gr.Slider(label="Sampling Steps (Recommended: 50)", minimum=1, maximum=100, step=1, value=50)
                    with gr.Row():
                        text_guide_scale = gr.Slider(label="text guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=3.0)
                        audio_guide_scale = gr.Slider(label="audio guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=5.0)
                    with gr.Row():
                        motion_frame = gr.Slider(label="motion frame", minimum=1, maximum=50, step=1, value=25)
                        fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=25)
                    with gr.Row():
                        overlap_window_length = gr.Slider(label="overlap window length", minimum=1, maximum=20, step=1, value=5)
                        seed_param = gr.Number(label="Seed, -1 for random", value=-1)
            with gr.Column():
                info = gr.Textbox(label="Info", interactive=False)
                video_output = gr.Video(label="Result", interactive=False)
                seed_output = gr.Textbox(label="Seed")

    gr.on(
        triggers=[generate_button.click, prompt.submit, negative_prompt.submit],
        fn = generate,
        inputs = [
            GPU_memory_mode,
            teacache_threshold,
            num_skip_start_steps,
            image_path,
            audio_path,
            prompt,
            negative_prompt,
            merge_audio_checkbox,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            text_guide_scale,
            audio_guide_scale,
            motion_frame,
            fps,
            overlap_window_length,
            seed_param,
        ],
        outputs = [video_output, seed_output, info]
    )
    exchange_button.click(
        fn=exchange_width_height,
        inputs=[width, height],
        outputs=[width, height, info]
    )
    adjust_button.click(
        fn=adjust_width_height,
        inputs=[image_path],
        outputs=[width, height, info]
    )


def extract_audio(video_file):
    """
    Extract audio from video file using audio_extractor.py
    """
    if not video_file:
        return None, "No video file provided."
    
    # Use a unique name to avoid conflicts
    output_filename = f"temp/extracted_audio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    
    # This is a subprocess call, which we are trying to phase out, but for these helpers it's acceptable for now.
    cmd = [
        "python", "audio_extractor.py",
        f"--video_path={video_file.name}",
        f"--saved_audio_path={output_filename}"
    ]
    
    try:
        print(f"Running audio extraction command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(output_filename):
            return output_filename, "Audio extracted successfully!"
        else:
            error_message = f"Error extracting audio. Stderr:\n{result.stderr}"
            print(error_message)
            return None, error_message
            
    except Exception as e:
        return None, f"Error extracting audio: {str(e)}"

def separate_vocals(audio_file):
    """
    Separate vocals from audio file using vocal_seperator.py
    """
    if not audio_file:
        return None, "No audio file provided."

    output_filename = f"temp/separated_vocal_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    
    # This is a subprocess call, which we are trying to phase out, but for these helpers it's acceptable for now.
    cmd = [
        "python", "vocal_seperator.py",
        f"--audio_separator_model_file=./checkpoints/Kim_Vocal_2.onnx",
        f"--audio_file_path={audio_file}",
        f"--saved_vocal_path={output_filename}"
    ]
    
    try:
        print(f"Running vocal separation command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(output_filename):
            return output_filename, "Vocal separation completed successfully!"
        else:
            error_message = f"Error separating vocals. Stderr:\n{result.stderr}"
            print(error_message)
            return None, error_message
            
    except Exception as e:
        return None, f"Error separating vocals: {str(e)}"


with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">StableAvatar</h2>
            </div>
            """)
    with gr.Accordion("Model Settings", open=False):
        with gr.Row():
            GPU_memory_mode = gr.Dropdown(
                label = "GPU Memory Mode",
                info = "Normal uses 25G VRAM, model_cpu_offload uses 13G VRAM",
                choices = ["Normal", "model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                value = "model_cpu_offload"
            )
            teacache_threshold = gr.Slider(label="teacache threshold", info = "Recommended: 0.1, 0 disables teacache acceleration", minimum=0, maximum=1, step=0.01, value=0)
            num_skip_start_steps = gr.Slider(label="Skip Start Steps", info = "Recommended: 5", minimum=0, maximum=100, step=1, value=5)
    
    with gr.TabItem("Inference"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_path = gr.Image(label="Upload Image", type="filepath", height=280)
                    audio_path = gr.Audio(label="Upload Audio", type="filepath")
                prompt = gr.Textbox(label="Prompt", value="")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="Low quality, bad quality, blur, blurry, (deformed iris, deformed pupils), (worst quality, low quality, normal quality), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck")
                generate_button = gr.Button("🎬 Generate", variant='primary')
                with gr.Accordion("Parameter Settings", open=True):
                    with gr.Row():
                        width = gr.Slider(label="Width", minimum=256, maximum=2048, step=16, value=512)
                        height = gr.Slider(label="Height", minimum=256, maximum=2048, step=16, value=512)
                    with gr.Row():
                        exchange_button = gr.Button("🔄 Swap Width/Height")
                        adjust_button = gr.Button("Adjust Width/Height based on Image")
                    with gr.Row():
                        guidance_scale = gr.Slider(label="guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=6.0)
                        num_inference_steps = gr.Slider(label="Sampling Steps (Recommended: 50)", minimum=1, maximum=100, step=1, value=50)
                    with gr.Row():
                        text_guide_scale = gr.Slider(label="text guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=3.0)
                        audio_guide_scale = gr.Slider(label="audio guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=5.0)
                    with gr.Row():
                        motion_frame = gr.Slider(label="motion frame", minimum=1, maximum=50, step=1, value=25)
                        fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=25)
                    with gr.Row():
                        overlap_window_length = gr.Slider(label="overlap window length", minimum=1, maximum=20, step=1, value=5)
                        seed_param = gr.Number(label="Seed, -1 for random", value=-1)
            with gr.Column():
                info = gr.Textbox(label="Info", interactive=False)
                video_output = gr.Video(label="Result", interactive=False)
                seed_output = gr.Textbox(label="Seed")
        
        gr.on(
            triggers=[generate_button.click, prompt.submit, negative_prompt.submit],
            fn = generate,
            inputs = [
                GPU_memory_mode,
                teacache_threshold,
                num_skip_start_steps,
                image_path,
                audio_path,
                prompt,
                negative_prompt,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                text_guide_scale,
                audio_guide_scale,
                motion_frame,
                fps,
                overlap_window_length,
                seed_param,
            ],
            outputs = [video_output, seed_output, info]
        )
        exchange_button.click(
            fn=exchange_width_height,
            inputs=[width, height],
            outputs=[width, height, info]
        )
        adjust_button.click(
            fn=adjust_width_height,
            inputs=[image_path],
            outputs=[width, height, info]
        )

    with gr.Tab("Audio Extraction"):
        with gr.Row():
            with gr.Column():
                video_input_extraction = gr.Video(label="Input Video")
                extract_button = gr.Button("Extract Audio")
                
            with gr.Column():
                extracted_audio_output = gr.Audio(label="Extracted Audio", type="filepath")
                extraction_status = gr.Textbox(label="Status", interactive=False)
                send_to_inference_audio = gr.Button("Send to Inference Tab")

        extract_button.click(
            fn=extract_audio,
            inputs=[video_input_extraction],
            outputs=[extracted_audio_output, extraction_status]
        )
        send_to_inference_audio.click(
            fn=lambda x: x,
            inputs=[extracted_audio_output],
            outputs=[audio_path]
        )

    with gr.Tab("Vocal Separation"):
        with gr.Row():
            with gr.Column():
                audio_input_separation = gr.Audio(type="filepath", label="Input Audio")
                separate_button = gr.Button("Separate Vocals")
                
            with gr.Column():
                vocal_output = gr.Audio(label="Vocal Output", type="filepath")
                separation_status = gr.Textbox(label="Status", interactive=False)
                send_to_inference_vocals = gr.Button("Send to Inference Tab")

        separate_button.click(
            fn=separate_vocals,
            inputs=[audio_input_separation],
            outputs=[vocal_output, separation_status]
        )
        send_to_inference_vocals.click(
            fn=lambda x: x,
            inputs=[vocal_output],
            outputs=[audio_path]
        )

if __name__ == "__main__":
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
    )
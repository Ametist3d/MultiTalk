import gradio as gr 
import os
'''
import torch

import shutil
import subprocess

import json
import tempfile
from huggingface_hub import snapshot_download
import soundfile as sf
from datetime import datetime
import signal
import threading

# Global variables
# Global process tracking
current_process = None
process_lock = threading.Lock()
VOLUME_PATH = "/workspace"
WEIGHTS_DIR = os.path.join(VOLUME_PATH, "weights")
num_gpus = torch.cuda.device_count()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'

def setup_directories():
    """Create necessary directories"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

def download_models():
    """Download all required models if not already present"""
    print(f"GPU AVAILABLE: {num_gpus}")
    
    # Define model paths
    wan_model_local_dir = os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P")
    wav2vec_local_dir = os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base")
    multitalk_local_dir = os.path.join(WEIGHTS_DIR, "MeiGen-MultiTalk")

    # Download Wan2.1-I2V-14B-480P model
    if not os.path.exists(wan_model_local_dir):
        print("Downloading Wan2.1-I2V-14B-480P model...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir=wan_model_local_dir,
        )
        print(f"Wan2.1-I2V-14B-480P model downloaded to: {wan_model_local_dir}")
    else:
        print(f"Wan2.1-I2V-14B-480P model already exists at: {wan_model_local_dir}")

    # Download Chinese wav2vec2 model
    if not os.path.exists(wav2vec_local_dir):
        print("Downloading chinese-wav2vec2-base model...")
        snapshot_download(
            repo_id="TencentGameMate/chinese-wav2vec2-base",
            local_dir=wav2vec_local_dir,
        )
        print(f"chinese-wav2vec2-base model downloaded to: {wav2vec_local_dir}")
    else:
        print(f"chinese-wav2vec2-base model already exists at: {wav2vec_local_dir}")

    # Download MeiGen MultiTalk weights
    if not os.path.exists(multitalk_local_dir):
        print("Downloading MeiGen-MultiTalk model...")
        snapshot_download(
            repo_id="MeiGen-AI/MeiGen-MultiTalk",
            local_dir=multitalk_local_dir,
        )
        print(f"MeiGen-MultiTalk model downloaded to: {multitalk_local_dir}")
    else:
        print(f"MeiGen-MultiTalk model already exists at: {multitalk_local_dir}")
    
    return wan_model_local_dir, wav2vec_local_dir, multitalk_local_dir

def setup_model_files(base_model_dir, multitalk_dir):
    """Setup model files by copying and renaming as needed"""
    # File to rename
    original_index = os.path.join(base_model_dir, "diffusion_pytorch_model.safetensors.index.json")
    backup_index = os.path.join(base_model_dir, "diffusion_pytorch_model.safetensors.index.json_old")

    # Rename the original index file (only if not already done)
    if os.path.exists(original_index) and not os.path.exists(backup_index):
        os.rename(original_index, backup_index)
        print("Renamed original index file to .json_old")

    # Copy updated index file from MultiTalk
    multitalk_index = os.path.join(multitalk_dir, "diffusion_pytorch_model.safetensors.index.json")
    target_index = os.path.join(base_model_dir, "diffusion_pytorch_model.safetensors.index.json")

    if os.path.exists(multitalk_index) and not os.path.exists(target_index):
        shutil.copy2(multitalk_index, base_model_dir)
        print("Copied MultiTalk index file")

    # Copy MultiTalk model weights
    multitalk_weights = os.path.join(multitalk_dir, "multitalk.safetensors")
    target_weights = os.path.join(base_model_dir, "multitalk.safetensors")

    if os.path.exists(multitalk_weights) and not os.path.exists(target_weights):
        shutil.copy2(multitalk_weights, base_model_dir)
        print("Copied MultiTalk model weights")

def detect_gpu_and_vram():
    """Detect GPU and calculate VRAM parameters"""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-compatible GPU found. An A100, L4 or L40S GPU is required.")
    
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Current GPU: {gpu_name}")

    if "A100" not in gpu_name and "L4" not in gpu_name and "L40S" not in gpu_name:
        print(f"Warning: This model is optimized for A100, L4, or L40S GPUs. Found: {gpu_name}")

    GPU_TO_VRAM_PARAMS = {
        "NVIDIA A100": 11000000000,
        "NVIDIA A100-SXM4-40GB": 11000000000,
        # "NVIDIA A100-SXM4-80GB": 22000000000,
        "NVIDIA A100-SXM4-80GB": 50000000000,
        "NVIDIA L4": 5000000000,
        "NVIDIA L40S": 11000000000,
        "NVIDIA L40": 15000000000,
        "L40": 15000000000,
        "RTX 5090": 15000000000,
    }

    used_vram_params = GPU_TO_VRAM_PARAMS.get(gpu_name, 15000000000) 
    print("Using", used_vram_params, "for num_persistent_param_in_dit")
    return used_vram_params

def trim_audio_to_5s_temp(audio_path, sample_rate=16000):
    """Trim audio to 5 seconds for shared UI"""
    max_duration_sec = 5
    audio, sr = sf.read(audio_path)

    if sr != sample_rate:
        sample_rate = sr

    max_samples = max_duration_sec * sample_rate
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    temp_filename = f"{base_name}_trimmed_{timestamp}.wav"
    temp_path = os.path.join(tempfile.gettempdir(), temp_filename)

    sf.write(temp_path, audio, samplerate=sample_rate)
    return temp_path

def create_temp_input_json(prompt: str, cond_image_path: str, cond_audio_path_spk1: str, cond_audio_path_spk2: str, n_prompt: str = "") -> str:
    """Create temporary JSON file with user inputs"""
    if cond_audio_path_spk2 is None:
        data = {
            "prompt": prompt,
            "n_prompt": n_prompt,
            "cond_image": cond_image_path,
            "cond_audio": {
                "person1": cond_audio_path_spk1
            }
        }
    else:
        data = {
            "prompt": prompt,
            "n_prompt": n_prompt,
            "cond_image": cond_image_path,
            "audio_type": "para",
            "cond_audio": {
                "person1": cond_audio_path_spk1,
                "person2": cond_audio_path_spk2
            }
        }

    temp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', encoding='utf-8')
    json.dump(data, temp_json, indent=4)
    temp_json_path = temp_json.name
    temp_json.close()

    print(f"Temporary input JSON saved to: {temp_json_path}")
    return temp_json_path

def load_prerendered_examples(prompt, cond_image_path, cond_audio_path_spk1, cond_audio_path_spk2, sample_steps):
    output_video = None
    
    if cond_image_path == "examples/single/single1.png":
        output_video = "examples/results/multitalk_single_example_1.mp4"
    elif cond_image_path == "examples/multi/3/multi3.png":
        output_video = "examples/results/multitalk_multi_example_2.mp4"

    return output_video
    
def build_command(input_json_path, sample_steps, frame_num, text_guide_scale, audio_guide_scale, 
                 seed, size, fps, use_apg, apg_momentum, apg_norm_threshold, teacache_thresh,
                 result_filename, wan_model_local_dir, wav2vec_local_dir, used_vram_params):
    """Build the command for inference"""
    common_args = [
        "--ckpt_dir", wan_model_local_dir,
        "--wav2vec_dir", wav2vec_local_dir,
        "--input_json", input_json_path,
        "--sample_steps", str(sample_steps),
        "--frame_num", str(frame_num),
        "--sample_text_guide_scale", str(text_guide_scale),
        "--sample_audio_guide_scale", str(audio_guide_scale),
        "--base_seed", str(seed),
        "--size", size,
        "--fps", str(fps),
        "--teacache_thresh", str(teacache_thresh),
        "--mode", "streaming",
        "--use_teacache",
        "--save_file", result_filename
    ]
    
    # Add APG if enabled
    if use_apg:
        common_args.extend([
            "--use_apg",
            "--apg_momentum", str(apg_momentum),
            "--apg_norm_threshold", str(apg_norm_threshold)
        ])

    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--standalone",
            "generate_multitalk.py",
            "--dit_fsdp", "--t5_fsdp",
            "--ulysses_size", str(num_gpus),
        ] + common_args
    else:
        cmd = [
            "python3", 
            "generate_multitalk.py",
            "--num_persistent_param_in_dit", str(used_vram_params),
        ] + common_args
    
    return cmd

def run_inference_non_streaming(cmd, result_filename):
    """Execute inference without streaming (for non-streaming infer function)"""
    global current_process
    
    try:
        log_content = ""
        
        with open("inference.log", "w") as log_file:
            with process_lock:
                current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
            
            # Read all output without yielding
            for line in current_process.stdout:
                print(line, end="")
                log_file.write(line)
                log_content += line
            
            current_process.wait()
        
        if current_process.returncode == 0:
            return f"{result_filename}.mp4", log_content
        else:
            return None, log_content + "\n[ERROR] Inference failed!"
            
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        with process_lock:
            current_process = None

def check_audio_duration(audio_path):
    """Get audio duration in seconds"""
    if not audio_path or not os.path.exists(audio_path):
        return 10.0
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        return len(audio) / sr
    except:
        return 10.0

def infer(prompt, cond_image_path, cond_audio_path_spk1, cond_audio_path_spk2, 
        sample_steps, frame_num, text_guide_scale, audio_guide_scale, seed, size, n_prompt,
        fps, use_apg, apg_momentum, apg_norm_threshold, teacache_thresh):
    """Main inference function (non-streaming version)"""
    
    # Validate audio duration early (same as streaming version)
    duration1 = check_audio_duration(cond_audio_path_spk1)
    duration2 = check_audio_duration(cond_audio_path_spk2) if cond_audio_path_spk2 else duration1
    min_duration = min(duration1, duration2)
    required_duration = frame_num / fps
    
    if min_duration < required_duration:
        error_msg = f"Audio too short: {min_duration:.1f}s available, {required_duration:.1f}s needed for {frame_num} frames at {fps}fps"
        return None, error_msg
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    result_filename = os.path.join(output_dir, f"MT_{sample_steps}stp_{text_guide_scale}tg_{audio_guide_scale}ag_{timestamp}")
    temp_files_to_cleanup = []
    
    try:
        # Create input JSON
        input_json_path = create_temp_input_json(prompt, cond_image_path, cond_audio_path_spk1, cond_audio_path_spk2, n_prompt)
        temp_files_to_cleanup.append(input_json_path)
        
        # Build and execute command
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        cmd = build_command(input_json_path, sample_steps, frame_num, text_guide_scale, 
            audio_guide_scale, seed, size, fps, use_apg, apg_momentum, 
            apg_norm_threshold, teacache_thresh, result_filename, 
            wan_model_local_dir, wav2vec_local_dir, used_vram_params)
        
        # Use non-streaming version
        video_path, logs = run_inference_non_streaming(cmd, result_filename)
        return video_path, logs

    except Exception as e:
        error_logs = f"Error during inference: {str(e)}"
        return None, error_logs
    
    finally:
        # Cleanup temporary files
        for f in temp_files_to_cleanup:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"[INFO] Removed temporary file: {f}")
            except Exception as e:
                print(f"[WARNING] Could not remove {f}: {e}")
                
def stop_inference():
    """Stop the current inference process"""
    global current_process
    with process_lock:
        if current_process:
            try:
                current_process.terminate()
                return "Inference stopped by user"
            except:
                return "Failed to stop inference"
        else:
            return "No inference running"

def format_log_line(line):
    return line[:80] + "..." if len(line) > 80 else line
    
#------------

def get_supported_image_files(folder_path):
    """Get list of supported image files from folder with detailed debugging"""
    print(f"DEBUG: Checking folder path: {folder_path}")
    
    if not folder_path or folder_path.strip() == "":
        return [], "No folder path provided"
    
    folder_path = folder_path.strip()
    
    if not os.path.exists(folder_path):
        return [], f"Folder does not exist: {folder_path}"
    
    if not os.path.isdir(folder_path):
        return [], f"Path is not a directory: {folder_path}"
    
    try:
        all_files = os.listdir(folder_path)
        print(f"DEBUG: Found {len(all_files)} total files in folder")
        print(f"DEBUG: Files: {all_files[:10]}...")  # Show first 10 files
    except PermissionError:
        return [], f"Permission denied accessing folder: {folder_path}"
    except Exception as e:
        return [], f"Error reading folder: {str(e)}"
    
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file.lower())[1]
            if ext in supported_exts:
                image_files.append(file_path)
    
    print(f"DEBUG: Found {len(image_files)} supported image files")
    if image_files:
        print(f"DEBUG: First few image files: {[os.path.basename(f) for f in image_files[:5]]}")
    
    status_msg = f"Found {len(image_files)} supported images out of {len(all_files)} total files"
    return sorted(image_files), status_msg

def validate_folder_path(folder_path):
    """Validate and provide info about folder path"""
    if not folder_path or folder_path.strip() == "":
        return "Please enter a folder path"
    
    folder_path = folder_path.strip()
    
    # Show current working directory for reference
    cwd = os.getcwd()
    
    if not os.path.exists(folder_path):
        return f"❌ Folder not found: {folder_path}\n💡 Current working directory: {cwd}\n💡 Try absolute paths like: /home/user/images or C:\\Users\\user\\images"
    
    if not os.path.isdir(folder_path):
        return f"❌ Path exists but is not a directory: {folder_path}"
    
    try:
        files = os.listdir(folder_path)
        image_files, status = get_supported_image_files(folder_path)
        
        return f"✅ Folder accessible: {folder_path}\n📁 Total files: {len(files)}\n🖼️ {status}\n💡 Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp"
    
    except PermissionError:
        return f"❌ Permission denied: {folder_path}\n💡 Check folder permissions"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def process_folder_batch(folder_path, prompt, cond_audio_path_spk1, cond_audio_path_spk2, 
                        sample_steps, frame_num, text_guide_scale, audio_guide_scale, seed, size, n_prompt,
                        fps, use_apg, apg_momentum, apg_norm_threshold, teacache_thresh):
    """Process all images in a folder"""
    
    image_files, status_msg = get_supported_image_files(folder_path)
    if not image_files:
        error_info = validate_folder_path(folder_path)
        return None, f"No supported image files found.\n\n{error_info}\n\nDEBUG INFO:\n{status_msg}"
    
    results = []
    all_logs = f"STARTING BATCH PROCESSING\n{status_msg}\n{'='*50}\n"
    successful_videos = []
    
    for i, image_path in enumerate(image_files):
        try:
            print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Generate unique seed for each image if original seed was -1
            current_seed = seed if seed != -1 else -1
            
            video_path, logs = infer(
                prompt, image_path, cond_audio_path_spk1, cond_audio_path_spk2,
                sample_steps, frame_num, text_guide_scale, audio_guide_scale, 
                current_seed, size, n_prompt, fps, use_apg, apg_momentum, 
                apg_norm_threshold, teacache_thresh
            )
            
            if video_path and os.path.exists(video_path):
                successful_videos.append(video_path)
                status = "✅ SUCCESS"
            else:
                status = "❌ FAILED"
            
            results.append({
                'image': os.path.basename(image_path),
                'status': status,
                'video': video_path
            })
            
            all_logs += f"\n{'='*50}\n"
            all_logs += f"Image {i+1}/{len(image_files)}: {os.path.basename(image_path)} - {status}\n"
            all_logs += f"{'='*50}\n"
            all_logs += logs + "\n"
            
        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            print(error_msg)
            all_logs += f"\n{'='*50}\n"
            all_logs += f"Image {i+1}/{len(image_files)}: {os.path.basename(image_path)} - ❌ ERROR\n"
            all_logs += error_msg + "\n"
            
            results.append({
                'image': os.path.basename(image_path),
                'status': "❌ ERROR", 
                'video': None
            })
    
    # Create summary
    total_files = len(image_files)
    successful_count = len(successful_videos)
    summary = f"""
BATCH PROCESSING COMPLETE
=========================
Total images: {total_files}
Successful: {successful_count}
Failed: {total_files - successful_count}

Generated Videos:
""" + "\n".join([f"• {os.path.basename(v)}" for v in successful_videos])
    
    all_logs = summary + "\n\n" + "DETAILED LOGS:\n" + all_logs
    
    # Return the first successful video for preview (or None if all failed)
    preview_video = successful_videos[0] if successful_videos else None
    
    return preview_video, all_logs
'''
def create_ui():

    with gr.Blocks(title="MultiTalk Inference") as demo:
        # Move CSS to the very top
        gr.HTML("""
        <style>
        * {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        html, body {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .gradio-container {
            padding: 0 !important;
            margin: 0 !important;
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Target all possible Gradio wrapper elements */
        .app, .main, .block, .container {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        
        /* Target the first row specifically */
        .gradio-container > div:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Remove gap between elements */
        .gap {
            gap: 0 !important;
        }
        
        /* Force remove all top spacing on rows */
        [data-testid="row"] {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        .custom-audio { max-height: 220px; }
        .logs-text textarea {
            font-family: 'Courier New', monospace !important;
            font-size: 10px !important;
            white-space: pre-wrap !important;
            word-break: break-all !important;
            overflow-wrap: break-word !important;
        }
        
        /* Hide Gradio footer */
        html { margin-bottom: -60px !important; }
        body::after {
            content: "";
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 40px;
            background: var(--neutral-950, #0f0f0f) !important;
            z-index: 99999;
        }
        </style>
        """)
        
        with gr.Row():
            with gr.Column(scale=10):
                pass  # Much more empty left side
            with gr.Column(scale=1, min_width=80):
                gr.Image("./assets/logo_V_long.png",
                        height=120, 
                        show_fullscreen_button=False,
                        show_download_button=False,
                        show_label=False, 
                        container=False)
        with gr.Row():
            # Left Column - Input Controls
            with gr.Column(scale=2):
                # Processing mode selection
                processing_mode = gr.Radio(
                    choices=["Single Image", "Batch Folder"],
                    value="Single Image",
                    label="Processing Mode"
                )

                # Single image input (visible by default)
                image_input = gr.Image(
                    type="filepath",
                    label="Conditioning Image",
                    height=302,
                    visible=True
                )

                # Folder input section (hidden by default)
                with gr.Group(visible=False) as folder_group:
                    folder_input = gr.Textbox(
                        label="Image Folder Path",
                        placeholder=f"Example: {os.path.join(os.getcwd(), 'images')}",
                        info="Enter absolute path to folder containing images"
                    )
                    
                    with gr.Row():
                        validate_btn = gr.Button("Validate Folder", size="sm")
                        current_dir_btn = gr.Button("Show Current Dir", size="sm")
                    
                    folder_info = gr.Textbox(
                        label="Folder Status", 
                        lines=4,
                        interactive=False,
                        elem_classes=["folder-info"]
                    )


                audio_input_spk1 = gr.Audio(
                    type="filepath",
                    label="Conditioning Audio for speaker 1(.wav)",
                    container=True,
                    scale=1,
                    elem_classes=["custom-audio"]
                )
                with gr.Accordion("Audio for speaker 2(.wav)", open=False):
                    audio_input_spk2 = gr.Audio(
                        type="filepath",
                        label="Conditioning Audio for speaker 2(.wav) (Optional)",
                        container=True,
                        scale=1,
                        elem_classes=["custom-audio"]
                    )

                with gr.Row():
                    submit_btn = gr.Button("Generate", variant="primary", size="lg")
                    

            # Middle Column - Settings
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Describe the scene...",
                    lines=2,
                )
                n_prompt = gr.Textbox(
                    label="Negative Prompt (Optional)",
                    placeholder="What you don't want to see...",
                    lines=2,
                    value="",
                    interactive=True
                )
                with gr.Accordion("Generation Settings", open=True):
                    with gr.Row():
                        sample_steps = gr.Slider(
                            label="Sample Steps",
                            value=12,
                            minimum=2,
                            maximum=25,
                            step=1
                        )
                        seed = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0
                        )
                    
                    with gr.Row():
                        fps = gr.Slider(
                            label="FPS",
                            value=25,
                            minimum=15,
                            maximum=30,
                            step=5,
                            info="Frames per second"
                        )
                        
                        frame_num = gr.Slider(
                            label="Frame Number (4n+1 format)",
                            value=81,
                            minimum=17,
                            maximum=161,
                            step=4
                        )
                    
                    # audio_info = gr.Textbox(
                    #     label="Audio Info",
                    #     value="Upload audio to see duration",
                    #     interactive=False
                    # )
                    with gr.Row():
                        text_guide_scale = gr.Slider(
                            label="Text Guidance Scale",
                            value=5.0,
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1
                        )
                        
                        audio_guide_scale = gr.Slider(
                            label="Audio Guidance Scale", 
                            value=4.0,
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1
                        )
                    

                    
                    size = gr.Dropdown(
                        label="Output Resolution",
                        choices=["multitalk-480", "multitalk-720"],
                        value="multitalk-480"
                    )
            
                with gr.Accordion("Advanced Settings", open=False):
                    use_apg = gr.Checkbox(
                        label="Enable APG (Adaptive Projected Guidance)",
                        value=False,
                        info="Improves guidance efficiency"
                    )
                    
                    with gr.Row(visible=False) as apg_controls:
                        apg_momentum = gr.Slider(
                            label="APG Momentum",
                            value=-0.75,
                            minimum=-1.0,
                            maximum=0.0,
                            step=0.05
                        )
                        
                        apg_norm_threshold = gr.Slider(
                            label="APG Norm Threshold",
                            value=55,
                            minimum=10,
                            maximum=100,
                            step=5
                        )
                    
                    teacache_thresh = gr.Slider(
                        label="TeaCache Threshold",
                        value=0.2,
                        minimum=0.1,
                        maximum=0.5,
                        step=0.05,
                        info="Higher = more aggressive caching"
                    )

                
                stop_btn = gr.Button("Stop", variant="stop", size="lg")

            # Right Column - Output and Examples
            with gr.Column(scale=3):
                # output_video = gr.Video(label="Generated Video", interactive=False, height=427, autoplay=True, show_download_button=True, container=True)
                with gr.Column(scale=3):
                    with gr.Group():
                        progress_bar = gr.Progress()
                        output_video = gr.Video(label="Generated Video", interactive=False, height=437, visible=True)            

                logs_output = gr.Textbox(
                    label="Generation Logs",
                    lines=12,
                    max_lines=12,
                    interactive=False,
                    show_copy_button=True,
                    container=True,
                    elem_classes=["logs-text"]
                )

                with gr.Accordion("Examples", open=False):
                    gr.Examples(
                        examples = [
                            ["A woman sings passionately in a dimly lit studio.", "examples/single/single1.png", "examples/single/1.wav", None, 12, 81, 5.0, 4.0, -1, "multitalk-480", ""],
                            ["In a cozy recording studio, a man and a woman are singing together. The man, with tousled brown hair, stands to the left, wearing a light green button-down shirt. His gaze is directed towards the woman, who is smiling warmly. She, with wavy dark hair, is dressed in a black floral dress and stands to the right, her eyes closed in enjoyment. Between them is a professional microphone, capturing their harmonious voices. The background features wooden panels and various audio equipment, creating an intimate and focused atmosphere. The lighting is soft and warm, highlighting their expressions and the intimate setting. A medium shot captures their interaction closely.", "examples/multi/3/multi3.png", "examples/multi/3/1-man.WAV", "examples/multi/3/1-woman.WAV", 12, 81, 5.0, 4.0, -1, "multitalk-480", ""],
                        ],
                        inputs = [prompt_input, image_input, audio_input_spk1, audio_input_spk2, sample_steps, frame_num, text_guide_scale, audio_guide_scale, seed, size, n_prompt, fps, use_apg, apg_momentum, apg_norm_threshold, teacache_thresh],
                    )

        # Event handlers
        def toggle_input_mode(mode):
            """Toggle between single image and folder input"""
            if mode == "Single Image":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        def show_current_directory():
            """Show current working directory"""
            cwd = os.getcwd()
            return f"Current working directory:\n{cwd}\n\nTo access local files:\n• Use absolute paths\n• Ensure folder is accessible from this location"

        def validate_folder_wrapper(folder_path):
            """Wrapper for folder validation"""
            return validate_folder_path(folder_path)

        def process_based_on_mode(mode, prompt, image_path, folder_path, audio1, audio2, 
                                sample_steps, frame_num, text_guide_scale, audio_guide_scale, 
                                seed, size, n_prompt, fps, use_apg, apg_momentum, apg_norm_threshold, teacache_thresh):
            """Process based on selected mode"""
            if mode == "Single Image":
                if not image_path:
                    return None, "Please select an image"
                return infer(prompt, image_path, audio1, audio2, sample_steps, frame_num, 
                           text_guide_scale, audio_guide_scale, seed, size, n_prompt,
                           fps, use_apg, apg_momentum, apg_norm_threshold, teacache_thresh)
            else:  # Batch Folder
                if not folder_path:
                    return None, "Please specify a folder path"
                return process_folder_batch(folder_path, prompt, audio1, audio2, 
                                          sample_steps, frame_num, text_guide_scale, audio_guide_scale, 
                                          seed, size, n_prompt, fps, use_apg, apg_momentum, 
                                          apg_norm_threshold, teacache_thresh)

        # Show/hide APG controls
        def toggle_apg_controls(use_apg):
            return gr.update(visible=use_apg)
        
        # Wire up event handlers
        processing_mode.change(toggle_input_mode, inputs=[processing_mode], outputs=[image_input, folder_group])
        use_apg.change(toggle_apg_controls, inputs=[use_apg], outputs=[apg_controls])
        validate_btn.click(validate_folder_wrapper, inputs=[folder_input], outputs=[folder_info])
        current_dir_btn.click(show_current_directory, outputs=[folder_info])
                
        submit_btn.click(
            # fn=process_based_on_mode,
            inputs=[
                processing_mode, prompt_input, image_input, folder_input, audio_input_spk1, audio_input_spk2, 
                sample_steps, frame_num, text_guide_scale, audio_guide_scale, 
                seed, size, n_prompt, fps, use_apg, apg_momentum, apg_norm_threshold, teacache_thresh
            ],
            outputs=[output_video, logs_output],
            show_progress=True
        )

        stop_btn.click(
            # fn=stop_inference,
            outputs=logs_output
        )

    return demo

def main():
    """Main function to initialize and run the app"""
    global wan_model_local_dir, wav2vec_local_dir, used_vram_params
    
    # Setup
    # setup_directories()
    # wan_model_local_dir, wav2vec_local_dir, multitalk_local_dir = download_models()
    # setup_model_files(wan_model_local_dir, multitalk_local_dir)
    # used_vram_params = detect_gpu_and_vram()
    
    print("Model setup completed.")
    
    # Create and launch UI
    demo = create_ui()
    demo.queue(
        max_size=4,
        default_concurrency_limit=1,
        api_open=False,
    ).launch(
        server_name="0.0.0.0", 
        server_port=8080, 
        ssr_mode=False, 
        show_error=True, 
        show_api=False,
        inbrowser=False,
        quiet=False
    )

if __name__ == "__main__":
    main()
import torch
import os
import shutil
import subprocess
import gradio as gr 
import json
import tempfile
from huggingface_hub import snapshot_download

import soundfile as sf
import tempfile
from datetime import datetime

is_shared_ui = True if "fffiloni/Meigen-MultiTalk" in os.environ['SPACE_ID'] else False

def trim_audio_to_5s_temp(audio_path, sample_rate=16000):
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

num_gpus = torch.cuda.device_count()
print(f"GPU AVAILABLE: {num_gpus}")

# Download All Required Models using `snapshot_download`

# Download Wan2.1-I2V-14B-480P model
wan_model_path = snapshot_download(
    repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
    local_dir="./weights/Wan2.1-I2V-14B-480P",
    #local_dir_use_symlinks=False
)

# Download Chinese wav2vec2 model
wav2vec_path = snapshot_download(
    repo_id="TencentGameMate/chinese-wav2vec2-base",
    local_dir="./weights/chinese-wav2vec2-base",
    #local_dir_use_symlinks=False
)

# Download MeiGen MultiTalk weights
multitalk_path = snapshot_download(
    repo_id="MeiGen-AI/MeiGen-MultiTalk",
    local_dir="./weights/MeiGen-MultiTalk",
    #local_dir_use_symlinks=False
)

# Define paths
base_model_dir = "./weights/Wan2.1-I2V-14B-480P"
multitalk_dir = "./weights/MeiGen-MultiTalk"

# File to rename
original_index = os.path.join(base_model_dir, "diffusion_pytorch_model.safetensors.index.json")
backup_index = os.path.join(base_model_dir, "diffusion_pytorch_model.safetensors.index.json_old")

# Rename the original index file
if os.path.exists(original_index):
    os.rename(original_index, backup_index)
    print("Renamed original index file to .json_old")

# Copy updated index file from MultiTalk
shutil.copy2(
    os.path.join(multitalk_dir, "diffusion_pytorch_model.safetensors.index.json"),
    base_model_dir
)

# Copy MultiTalk model weights
shutil.copy2(
    os.path.join(multitalk_dir, "multitalk.safetensors"),
    base_model_dir
)

print("Copied MultiTalk files into base model directory.")



# Check if CUDA-compatible GPU is available
if torch.cuda.is_available():
    # Get current GPU name
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Current GPU: {gpu_name}")

    # Enforce GPU requirement
    if "A100" not in gpu_name and "L4" not in gpu_name:
        raise RuntimeError(f"This notebook requires an A100 or L4 GPU. Found: {gpu_name}")
    elif "L4" in gpu_name:
        print("Warning: L4 or L40S is supported, but A100 is recommended for faster inference.")
else:
    raise RuntimeError("No CUDA-compatible GPU found. An A100, L4 or L40S GPU is required.")


GPU_TO_VRAM_PARAMS = {
    "NVIDIA A100": 11000000000,
    "NVIDIA A100-SXM4-40GB": 11000000000,
    "NVIDIA A100-SXM4-80GB": 22000000000,
    "NVIDIA L4": 5000000000,
    "NVIDIA L40S": 11000000000
}
USED_VRAM_PARAMS = GPU_TO_VRAM_PARAMS[gpu_name]
print("Using", USED_VRAM_PARAMS, "for num_persistent_param_in_dit")



def create_temp_input_json(prompt: str, cond_image_path: str, cond_audio_path_spk1: str, cond_audio_path_spk2: str) -> str:
    """
    Create a temporary JSON file with the user-provided prompt, image, and audio paths.
    Returns the path to the temporary JSON file.
    """
    # Structure based on your original JSON format
    if cond_audio_path_spk2 is None:
        data = {
            "prompt": prompt,
            "cond_image": cond_image_path,
            "cond_audio": {
                "person1": cond_audio_path_spk1
            }
        }

    else:
        data = {
            "prompt": prompt,
            "cond_image": cond_image_path,
            "audio_type": "para",
            "cond_audio": {
                "person1": cond_audio_path_spk1,
                "person2": cond_audio_path_spk2
            }
        }

    # Create a temp file
    temp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', encoding='utf-8')
    json.dump(data, temp_json, indent=4)
    temp_json_path = temp_json.name
    temp_json.close()

    print(f"Temporary input JSON saved to: {temp_json_path}")
    return temp_json_path


def infer(prompt, cond_image_path, cond_audio_path_spk1, cond_audio_path_spk2, sample_steps):

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    result_filename = f"meigen_multitalk_result_{sample_steps}_steps_{timestamp}"
    temp_files_to_cleanup = []
    
    if is_shared_ui:
        trimmed_audio_path_spk1 = trim_audio_to_5s_temp(cond_audio_path_spk1)
        if trimmed_audio_path_spk1 != cond_audio_path_spk1:
            cond_audio_path_spk1 = trimmed_audio_path_spk1
            temp_files_to_cleanup.append(trimmed_audio_path_spk1)

        if cond_audio_path_spk2 is not None:
            trimmed_audio_path_spk2 = trim_audio_to_5s_temp(cond_audio_path_spk2)
            if trimmed_audio_path_spk2 != cond_audio_path_spk2:
                cond_audio_path_spk2 = trimmed_audio_path_spk2
                temp_files_to_cleanup.append(trimmed_audio_path_spk2)

    # Prepare input JSON
    input_json_path = create_temp_input_json(prompt, cond_image_path, cond_audio_path_spk1, cond_audio_path_spk2)
    temp_files_to_cleanup.append(input_json_path)
    
# Base args
    common_args = [
        "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", "weights/chinese-wav2vec2-base",
        "--input_json", input_json_path,
        "--sample_steps", str(sample_steps),
        "--mode", "streaming",
        "--use_teacache",
        "--save_file", result_filename
    ]

    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--standalone",
            "generate_multitalk.py",
            #"--num_persistent_param_in_dit", "22000000000", # On 4xL40S
            "--dit_fsdp", "--t5_fsdp",
            "--ulysses_size", str(num_gpus),
        ] + common_args
    else:
        cmd = [
            "python3", 
            "generate_multitalk.py",
            "--num_persistent_param_in_dit", str(USED_VRAM_PARAMS),
        ] + common_args

    try:
        # Log to file and stream
        with open("inference.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
            process.wait()

        if process.returncode != 0:
            raise RuntimeError("Inference failed. Check inference.log for details.")

        return f"{result_filename}.mp4"

    finally:
        for f in temp_files_to_cleanup:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"[INFO] Removed temporary file: {f}")
            except Exception as e:
                print(f"[WARNING] Could not remove {f}: {e}")  

def load_prerendered_examples(prompt, cond_image_path, cond_audio_path_spk1, cond_audio_path_spk2, sample_steps):
    output_video = None
    
    if cond_image_path == "examples/single/single1.png":
        output_video = "examples/results/multitalk_single_example_1.mp4"
    elif cond_image_path == "examples/multi/3/multi3.png":
        output_video = "examples/results/multitalk_multi_example_2.mp4"

    return output_video

with gr.Blocks(title="MultiTalk Inference") as demo:
    gr.Markdown("## 🎤 Meigen MultiTalk Inference Demo")
    gr.Markdown("Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation")
    if is_shared_ui:
        gr.Markdown("Audio will be trimmed to max 5 seconds on fffiloni's shared UI. Sample steps are limited to 12. Gradio queue size is set to 4. Generating a 5 seconds video will take approximatively 20 minutes. Duplicate to skip the queue and work with longer audio inference. ")
    gr.HTML("""
    <div style="display:flex;column-gap:4px;">
        <a href="https://github.com/MeiGen-AI/MultiTalk">
            <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
        </a>
        <a href='https://meigen-ai.github.io/multi-talk/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
        <a href='https://huggingface.co/MeiGen-AI/MeiGen-MultiTalk'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
        <a href='https://arxiv.org/abs/2505.22647'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
        <a href="https://huggingface.co/spaces/fffiloni/Meigen-MultiTalk?duplicate=true">
            <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-sm.svg" alt="Duplicate this Space">
        </a>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Text Prompt",
                placeholder="Describe the scene...",
            )

            image_input = gr.Image(
                type="filepath",
                label="Conditioning Image"
            )

            audio_input_spk1 = gr.Audio(
                type="filepath",
                label="Conditioning Audio for speaker 1(.wav)"
            )

            audio_input_spk2 = gr.Audio(
                type="filepath",
                label="Conditioning Audio for speaker 2(.wav) (Optional)"
            )

            with gr.Accordion("Advanced settings", open=False):
                sample_steps = gr.Slider(
                    label="sample steps",
                    value=12,
                    minimum=2,
                    maximum=25,
                    step=1,
                    interactive=False if is_shared_ui else True
                )

            submit_btn = gr.Button("Generate")

        with gr.Column(scale=3):
            output_video = gr.Video(label="Generated Video", interactive=False)

            gr.Examples(
                examples = [
                    ["A woman sings passionately in a dimly lit studio.", "examples/single/single1.png", "examples/single/1.wav", None, 12, "examples/results/multitalk_single_example_1.mp4"],
                    ["In a cozy recording studio, a man and a woman are singing together. The man, with tousled brown hair, stands to the left, wearing a light green button-down shirt. His gaze is directed towards the woman, who is smiling warmly. She, with wavy dark hair, is dressed in a black floral dress and stands to the right, her eyes closed in enjoyment. Between them is a professional microphone, capturing their harmonious voices. The background features wooden panels and various audio equipment, creating an intimate and focused atmosphere. The lighting is soft and warm, highlighting their expressions and the intimate setting. A medium shot captures their interaction closely.", "examples/multi/3/multi3.png", "examples/multi/3/1-man.WAV", "examples/multi/3/1-woman.WAV", 12, "examples/results/multitalk_multi_example_2.mp4"],
                ],
                inputs = [prompt_input, image_input, audio_input_spk1, audio_input_spk2, sample_steps, output_video],
            )

    submit_btn.click(
        fn=infer,
        inputs=[prompt_input, image_input, audio_input_spk1, audio_input_spk2, sample_steps],
        outputs=output_video
    )

demo.queue(max_size=4).launch(ssr_mode=False, show_error=True, show_api=False)
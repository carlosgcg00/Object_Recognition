import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

# Title and Intro
nb.cells.append(new_markdown_cell("""# Video Restoration Pipeline (Real-ESRGAN + RIFE)

This notebook automates a video restoration pipeline. It upscales videos to 4K using Real-ESRGAN and interpolates them to 60fps using RIFE. It manages GPU VRAM via tiling and exports an H.264 encoded MP4 video.

**Instructions:**
1. Define your `input_path` and `output_path` in the "User Configuration" cell.
2. Run all cells (`Runtime -> Run all`)."""))

# Configure notebook metadata for Colab GPU
nb.metadata = {
    "accelerator": "GPU",
    "colab": {
        "gpuType": "all",
        "provenance": []
    },
    "kernelspec": {
        "display_name": "Python 3",
        "name": "python3"
    },
    "language_info": {
        "name": "python"
    }
}

# Cell 1: Environment Setup
nb.cells.append(new_markdown_cell("""## 1. Environment Setup
Install dependencies, clone repositories, download models, and apply necessary patches for Colab."""))

nb.cells.append(new_code_cell("""# @title Setup Environment (Run Once)
import os
import sys

# Install missing dependency
!pip install sk-video

# Clone Real-ESRGAN
!git clone https://github.com/xinntao/Real-ESRGAN.git
%cd Real-ESRGAN
# Set up environment
!pip install basicsr facexlib gfpgan
!pip install -r requirements.txt
!python setup.py develop

# Fix the basicsr compatibility issue
try:
    import basicsr
    # Dynamically find the basicsr installation path
    basicsr_path = os.path.dirname(basicsr.__file__)
    file_path = os.path.join(basicsr_path, 'data', 'degradations.py')

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()

        # Replace the outdated import with the modern one
        updated_content = content.replace(
            'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
            'from torchvision.transforms.functional import rgb_to_grayscale'
        )

        with open(file_path, 'w') as f:
            f.write(updated_content)
        print(f"Successfully patched {file_path}")
    else:
        print(f"Degradations file not found at {file_path}")
except ImportError:
    print("Could not import basicsr to apply the patch. Ensure it installed correctly.")

# Download RealESRGAN x4plus model
!wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
%cd ..

# Clone ECCV2022-RIFE
!git clone https://github.com/hzwer/ECCV2022-RIFE.git
%cd ECCV2022-RIFE
# RIFE dependencies
!pip install opencv-python
# Download RIFE model
import urllib.request
import zipfile
url = "https://drive.google.com/uc?export=download&id=1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_"
import gdown
!gdown $url -O RIFE_model.zip
!unzip -o RIFE_model.zip -d train_log
%cd ..

!apt-get install -y ffmpeg"""))

# Cell 2: User Configuration
nb.cells.append(new_markdown_cell("""## 2. User Configuration
Define the input and output video paths."""))

nb.cells.append(new_code_cell("""# @title Define Paths
input_path = "input_video.mp4" # @param {type:"string"}
output_path = "restored_video.mp4" # @param {type:"string"}

import os
# Ensure input exists
if not os.path.exists(input_path):
    print(f"Warning: input_path '{input_path}' does not exist! Please upload it or change the path.")"""))

# Cell 3: Extract Frames & Audio
nb.cells.append(new_markdown_cell("""## 3. Extract Frames and Audio
Extract frames to apply frame-by-frame processing."""))

nb.cells.append(new_code_cell("""# @title Extract Frames
!mkdir -p workspace/frames
!mkdir -p workspace/audio
!rm -rf workspace/frames/*

# Extract audio
!ffmpeg -y -i "{input_path}" -vn -acodec copy workspace/audio/audio.aac

# Extract frames
!ffmpeg -y -i "{input_path}" -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 workspace/frames/frame_%08d.jpg

# Get original FPS
import subprocess
import shlex

cmd = f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate \\\"{input_path}\\\""
try:
    fps_raw = subprocess.check_output(shlex.split(cmd)).decode('utf-8').strip()
    if '/' in fps_raw:
        num, den = fps_raw.split('/')
        orig_fps = float(num) / float(den)
    else:
        orig_fps = float(fps_raw)
except Exception as e:
    orig_fps = 30.0
    print(f"Could not determine FPS, defaulting to {orig_fps}")

print(f"Original FPS: {orig_fps}")"""))

# Cell 4: Upscale with Real-ESRGAN
nb.cells.append(new_markdown_cell("""## 4. Upscale with Real-ESRGAN
Upscale the frames. Tiling is enabled to prevent Out-Of-Memory (OOM) errors on the GPU."""))

nb.cells.append(new_code_cell("""# @title Upscale Frames
!mkdir -p workspace/upscaled_frames
!rm -rf workspace/upscaled_frames/*

# We use tiling (-t 512) to avoid OOM
!python Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x4plus -i workspace/frames -o workspace/upscaled_frames --outscale 4 --ext jpg -t 512 --face_enhance"""))

# Cell 5: Interpolate with RIFE
nb.cells.append(new_markdown_cell("""## 5. Interpolate to 60fps with RIFE"""))

nb.cells.append(new_code_cell("""# @title Interpolate Frames
!mkdir -p workspace/interpolated_frames
!rm -rf workspace/interpolated_frames/*

# Assuming you want to double the framerate. If original is ~30fps, exp=1 gives ~60fps.
# If original is 24fps, exp=1 gives 48fps, exp=2 gives 96fps.
# Let's use exp=1 for 2x framerate. We process frames directly.
# Wait, ECCV2022-RIFE inference_video.py processes video. inference_img.py processes 2 images.
# Actually, it's easier to encode the upscaled frames into a temporary video and pass that to RIFE.
!ffmpeg -y -framerate {orig_fps} -i workspace/upscaled_frames/frame_%08d_out.jpg -c:v libx264 -crf 18 workspace/temp_upscaled.mp4

# Now run RIFE on the temporary upscaled video
# --exp=1 means 2x interpolation. --scale=0.5 recommended for 4K to avoid OOM in optical flow.
# Output will be something like temp_upscaled_2X_...mp4
!python ECCV2022-RIFE/inference_video.py --exp=1 --video workspace/temp_upscaled.mp4 --scale=0.5

# Find the output video of RIFE
import glob
rife_outputs = glob.glob("workspace/temp_upscaled_2X_*.mp4")
if rife_outputs:
    rife_video = rife_outputs[0]
else:
    rife_video = "workspace/temp_upscaled.mp4"
    print("RIFE failed or skipped, using upscaled video instead.")"""))

# Cell 6: Final Export
nb.cells.append(new_markdown_cell("""## 6. Final Export
Combine the interpolated video with the original audio and export to `output_path` using H.264."""))

nb.cells.append(new_code_cell("""# @title Combine and Export
import os

# Check if we have audio
has_audio = os.path.exists("workspace/audio/audio.aac") and os.path.getsize("workspace/audio/audio.aac") > 0

if has_audio:
    # Combine video and audio
    !ffmpeg -y -i "{rife_video}" -i workspace/audio/audio.aac -c:v libx264 -crf 18 -c:a aac -b:a 192k -strict experimental "{output_path}"
else:
    # Just encode video
    !ffmpeg -y -i "{rife_video}" -c:v libx264 -crf 18 "{output_path}"

print(f"Done! Video saved to {output_path}")"""))

# Write out the notebook
with open('notebook_scripts/video_restoration_pipeline.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("Notebook generated successfully at notebook_scripts/video_restoration_pipeline.ipynb")

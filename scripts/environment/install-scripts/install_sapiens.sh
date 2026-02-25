#! /bin/bash

############################
# Sapiens pose extractor
############################

module purge
module load v100
module load cuda/12.9.1
module load miniforge3/25.3.0-3

# Resolve paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/../../.." )"

venvs="$PROJECT_ROOT/venvs"
tools="$PROJECT_ROOT/tools"
repo_root="$tools/sapiens"
repo_dir="$repo_root/Sapiens-Pytorch-Inference"

mkdir -p "$repo_root"

# Activate env
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$venvs/sapiens_pose_extractor"

echo "Using env: $venvs/sapiens_pose_extractor"

export PYTHONNOUSERSITE=1

# -------------------------
# Install PyTorch (CUDA)
# -------------------------
echo "Installing PyTorch (CUDA 12.1 wheels)..."

pip install \
  torch==2.5.1+cu121 \
  torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121


# -------------------------
# Clone repository
# -------------------------
if [[ ! -d "$repo_dir/.git" ]]; then
    git clone https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference.git "$repo_dir"
fi

cd "$repo_dir"
git fetch --all
git checkout a0bec2e524ef825fb269b691b32353426e93b5d0

# -------------------------
# Install requirements (excluding torch)
# -------------------------
grep -vE "^(torch|torchvision)" requirements.txt > /tmp/req_no_torch.txt
pip install -r /tmp/req_no_torch.txt

# Extra deps required for inference
pip install opencv-python huggingface_hub onnx onnxruntime

# Install package
pip install -e .

# -------------------------
# Download model weights
# -------------------------

echo "Downloading Sapiens and YOLOv8 models..."

python <<PYTHON
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

repo_dir = Path("$repo_dir")
models_dir = repo_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

def copy_if_missing(src, dst):
    if not dst.exists():
        shutil.copy(src, dst)

# -------------------------
# Sapiens pose model
# -------------------------
sapiens_file = "sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt2"
sapiens_target = models_dir / sapiens_file

if not sapiens_target.exists():
    print("Downloading Sapiens pose model...")
    src = hf_hub_download(
        repo_id="facebook/sapiens-pose-1b-torchscript",
        filename=sapiens_file,
        revision="4caa2b2290255dc8963b5ead35fe3c6e761742aa"
    )
    copy_if_missing(src, sapiens_target)
else:
    print("Sapiens model already present.")

# -------------------------
# YOLOv8m checkpoint
# -------------------------
yolo_file = "yolov8m.pt"
yolo_target = models_dir / yolo_file

if not yolo_target.exists():
    print("Downloading YOLOv8m model...")
    src = hf_hub_download(
        repo_id="Ultralytics/YOLOv8",
        filename=yolo_file,
        revision="8a9e1a55f987a77f9966c2ac3f80aa8aa37b3c1a"
    )
    copy_if_missing(src, yolo_target)
else:
    print("YOLOv8m already present.")

print("Models ready.")
PYTHON



echo "✅ Sapiens pose extractor ready."
echo "Activate with: conda activate $venvs/sapiens_pose_extractor"
echo "Repo path: $repo_dir"

conda deactivate

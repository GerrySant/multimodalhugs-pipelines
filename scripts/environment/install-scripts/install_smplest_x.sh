#! /bin/bash

############################
# SMPLest-X pose extractor (py38)
############################

module purge
module load a100
module load cuda/11.8.0
module load miniforge3/25.3.0-3

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "SCRIPT_DIR: $SCRIPT_DIR"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/../../.." )"
echo "PROJECT_ROOT: $PROJECT_ROOT"

venvs="$PROJECT_ROOT/venvs"
tools="$PROJECT_ROOT/tools"
repo_dir="$tools/smplest-x"

mkdir -p "$tools"

# Activate conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$venvs/smplest_x_pose_extractor"

echo "Using env: $venvs/smplest_x_pose_extractor"

# -------------------------
# Install PyTorch
# -------------------------
mamba install -y \
    pytorch==1.12.0 \
    torchvision==0.13.0 \
    torchaudio==0.12.0 \
    cudatoolkit=11.3 \
    -c pytorch

# -------------------------
# Clone repository (SSH)
# -------------------------
if [[ ! -d "$repo_dir/.git" ]]; then
    git clone git@github.com:GerrySant/SMPLest-X.git "$repo_dir"
fi

cd "$repo_dir"
git fetch --all
git checkout pose_estimation_study

# -------------------------
# Install python requirements
# -------------------------
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Downloading pretrained models..."

PRETRAIN_DIR="$repo_dir/pretrained_models"
SMPL_DIR="$PRETRAIN_DIR/smplest_x_h"

mkdir -p "$SMPL_DIR"

python <<PYTHON
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

repo_root = Path("$repo_dir")
pretrain_dir = repo_root / "pretrained_models"
smpl_dir = pretrain_dir / "smplest_x_h"
human_dir = repo_root / "human_models" / "smplx"

pretrain_dir.mkdir(parents=True, exist_ok=True)
smpl_dir.mkdir(parents=True, exist_ok=True)
human_dir.mkdir(parents=True, exist_ok=True)

def copy_if_missing(src, dst):
    if not dst.exists():
        shutil.copy(src, dst)

# -------------------------
# Download checkpoint
# -------------------------
ckpt = hf_hub_download(
    repo_id="waanqii/SMPLest-X",
    filename="smplest_x_h.pth.tar"
)

target_ckpt = smpl_dir / "smplest_x_h.pth.tar"
if not target_ckpt.exists():
    shutil.copy(ckpt, target_ckpt)

# -------------------------
# Download config
# -------------------------
config_src = hf_hub_download(
    repo_id="waanqii/SMPLest-X",
    filename="config_base.py"
)

target_config = smpl_dir / "config_base.py"
shutil.copy(config_src, target_config)

# -------------------------
# PATCH human_model_path
# -------------------------
text = target_config.read_text()

text = text.replace(
    "'./human_models/human_model_files'",
    "'./human_models/'"
)

target_config.write_text(text)

# -------------------------
# YOLOv8 weights
# -------------------------
yolo = hf_hub_download(
    repo_id="Ultralytics/YOLOv8",
    filename="yolov8x.pt"
)

target_yolo = pretrain_dir / "yolov8x.pt"
if not target_yolo.exists():
    shutil.copy(yolo, target_yolo)

# -------------------------
# SMPL-X human models
# -------------------------
files = [
    "SMPLX_FEMALE.npz",
    "SMPLX_FEMALE.pkl",
    "SMPLX_MALE.npz",
    "SMPLX_MALE.pkl",
    "SMPLX_NEUTRAL.npz",
    "SMPLX_NEUTRAL.pkl",
]

for fname in files:
    src = hf_hub_download(
        repo_id="lilpotat/pytorch3d",
        filename=f"models/{fname}"
    )
    copy_if_missing(src, human_dir / fname)

print("All pretrained assets ready.")
PYTHON

conda install -y -c conda-forge libglu
conda install -y -c conda-forge mesalib

echo "✅ SMPLest-X environment ready."
echo "Activate with: conda activate $venvs/smplest_x_pose_extractor"
echo "Repo path: $repo_dir"

conda deactivate
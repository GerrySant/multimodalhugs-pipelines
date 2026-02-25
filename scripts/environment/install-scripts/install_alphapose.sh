#! /bin/bash

############################
# AlphaPose pose extractor (py310)
############################
module purge
module load v100 
#module load cuda/11.8.0 
module load cuda/12.6.3
module load miniforge3/25.3.0-3

# Resolve paths like in your other installers
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "SCRIPT_DIR: $SCRIPT_DIR"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/../../.." )"
echo "PROJECT_ROOT: $PROJECT_ROOT"
venvs="$PROJECT_ROOT/venvs"
echo "venvs: $venvs"

# Where to place external repos/tools
tools="$PROJECT_ROOT/tools"
alphapose_dir="$tools/alphapose/AlphaPose"
mkdir -p "$tools/alphapose"

# Activate env (prefix)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $venvs/alphapose_pose_extractor

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/../.." )"
venvs="$PROJECT_ROOT/venvs"

# CUDA env vars (needed for compiling some deps / extensions)
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc || true)")")"
if [[ -d "${CUDA_HOME:-}/bin" ]]; then
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export PIP_NO_BUILD_ISOLATION=1
fi

# Avoid build isolation so legacy builds use env's numpy/cython
export PIP_NO_BUILD_ISOLATION=1

echo "CUDA_HOME: ${CUDA_HOME:-<not found>}"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"

# -------------------------
# Torch (CUDA 11.8, like your known-working setup)
# If you *must* match CUDA 13 modules, tell me what torch/cuda combo you want and I adapt it.
# -------------------------
#mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


# -------------------------
# Pin pip & setuptools (known-good for setup.py develop flows)
# -------------------------
python -m pip install --upgrade "pip<25" "setuptools<70" wheel

# -------------------------
# Clone AlphaPose + pin commit (your exact commit)
# -------------------------
if [[ ! -d "$alphapose_dir/.git" ]]; then
  git clone git@github.com:MVIG-SJTU/AlphaPose.git "$alphapose_dir"
fi

cd "$alphapose_dir"
git fetch --all --tags
git checkout "c60106d19afb443e964df6f06ed1842962f5f1f7"

# -------------------------
# Python deps (pinned similarly to your working recipe)
# - numpy<2 avoids legacy np.float issues
# - opencv pinned to avoid pulling numpy>=2
# -------------------------
python -m pip install \
  "numpy==1.26.4" \
  "opencv-python==4.11.0.86" \
  "matplotlib==3.10.7" \
  "scipy==1.15.3" \
  "tqdm==4.67.1" \
  "tensorboardx==2.6.4" \
  "visdom==0.2.4" \
  "terminaltables==3.1.10" \
  "easydict==1.13" \
  "six==1.17.0" \
  "natsort==8.4.0" \
  "timm==0.1.20" \
  "pyyaml==6.0.3" \
  "protobuf==6.33.0" \
  "packaging==25.0" \
  "cython==3.2.0"

python -m pip install "pycocotools==2.0.10" "halpecocotools==0.0.1"

# cython_bbox (version that compiled & ran for you)
export PIP_NO_BUILD_ISOLATION=1
python -m pip install --no-build-isolation "cython-bbox==0.1.5"

# -------------------------
# Build AlphaPose (develop)
# -------------------------
export PIP_NO_BUILD_ISOLATION=1
python setup.py build develop

python -m pip install gdown

# -------------------------
# Download YOLOv3-SPP weights (required by AlphaPose detector)
# -------------------------

YOLO_DIR="$alphapose_dir/detector/yolo/data"
YOLO_WEIGHTS="$YOLO_DIR/yolov3-spp.weights"
YOLO_URL="https://pjreddie.com/media/files/yolov3-spp.weights"

mkdir -p "$YOLO_DIR"

if [[ ! -f "$YOLO_WEIGHTS" ]]; then
  echo "Downloading YOLOv3-SPP weights..."
  echo "Destination: $YOLO_WEIGHTS"

  # try curl first, fallback to wget
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$YOLO_WEIGHTS" "$YOLO_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$YOLO_WEIGHTS" "$YOLO_URL"
  else
    echo "ERROR: neither curl nor wget is available" >&2
    exit 1
  fi
else
  echo "YOLO weights already present, skipping download."
fi

# sanity check (file should be ~248MB)
if [[ -f "$YOLO_WEIGHTS" ]]; then
  size=$(stat -c%s "$YOLO_WEIGHTS")
  if [[ "$size" -lt 200000000 ]]; then
    echo "ERROR: downloaded weights look incomplete ($size bytes)" >&2
    exit 1
  fi
fi

# -------------------------
# Download AlphaPose SPPE checkpoint
# -------------------------

AP_MODEL_DIR="$alphapose_dir/pretrained_models"
AP_MODEL_FILE="multi_domain_fast50_dcn_combined_256x192.pth"
AP_MODEL_PATH="$AP_MODEL_DIR/$AP_MODEL_FILE"
AP_MODEL_ID="1wX1Z2ZOoysgSNovlgiEtJKpbR8tUBWYR"
AP_MODEL_ZOO_URL="https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md#multi-domain-models-strongly-recommended"

manual_download_instructions() {
  echo "ERROR: Could not download AlphaPose checkpoint automatically." >&2
  echo "" >&2
  echo "Please download it manually from:" >&2
  echo "  $AP_MODEL_ZOO_URL" >&2
  echo "" >&2
  echo "In that table, download the model from the *second row* (136 keypoints):" >&2
  echo "  multi_domain_fast50_dcn_combined_256x192.pth" >&2
  echo "" >&2
  echo "Then place it at:" >&2
  echo "  $AP_MODEL_PATH" >&2
}

mkdir -p "$AP_MODEL_DIR"

if [[ ! -f "$AP_MODEL_PATH" ]]; then
  echo "Downloading AlphaPose checkpoint..."
  echo "Destination: $AP_MODEL_PATH"

  # Ensure gdown exists
  if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found. Installing it..." >&2
    python -m pip install -q gdown || { manual_download_instructions; exit 1; }
  fi

  # Try download
  gdown --fuzzy "https://drive.google.com/file/d/$AP_MODEL_ID/view" -O "$AP_MODEL_PATH"
  rc=$?

  if [[ $rc -ne 0 ]]; then
    rm -f "$AP_MODEL_PATH"
    manual_download_instructions
    exit 1
  fi
else
  echo "AlphaPose checkpoint already present, skipping."
fi

# sanity check (~102MB expected)
if [[ -f "$AP_MODEL_PATH" ]]; then
  size=$(stat -c%s "$AP_MODEL_PATH" 2>/dev/null || echo 0)
  if [[ "$size" -lt 50000000 ]]; then
    echo "ERROR: checkpoint file looks incomplete ($size bytes): $AP_MODEL_PATH" >&2
    rm -f "$AP_MODEL_PATH"
    manual_download_instructions
    exit 1
  fi
fi

echo "✅ AlphaPose pose extractor environment ready."
echo "Activate with:  source activate $venvs/alphapose_pose_extractor"
echo "Repo path:      $alphapose_dir"

conda deactivate

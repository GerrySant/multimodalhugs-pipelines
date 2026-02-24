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

echo "✅ AlphaPose pose extractor environment ready."
echo "Activate with:  source activate $venvs/alphapose_pose_extractor"
echo "Repo path:      $alphapose_dir"

conda deactivate

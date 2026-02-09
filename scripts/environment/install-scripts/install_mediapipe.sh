#! /bin/bash

############################
# mediapipe holistic (py311)
############################

module load gpumem32gb cuda/13.0.2 cudnn/9.8.0.87-12 miniforge3

environment_scripts=$(dirname "$0")
scripts=$environment_scripts/../..
base=$scripts/..

venvs=$base/venvs

############################
# mediapipe holistic (py39)
############################
source activate $venvs/mediapipe
tools=$base/tools/mediapipe
mkdir -p $tools

pip install pose-format
pip install "mediapipe<0.10.30"

# install multimodalhugs

git clone https://github.com/GerrySant/multimodalhugs.git $tools/multimodalhugs

# pin commit  https://github.com/GerrySant/multimodalhugs/commit/5201c80f27aa70c460e8297a799dc5daccbd1b3b
# to avoid unintentionally breaking the code

(cd $tools/multimodalhugs && git checkout "5201c80f27aa70c460e8297a799dc5daccbd1b3b")

(cd $tools/multimodalhugs && pip install .)

# TF keras, because keras 3 is not supported in Transformers

pip install tf-keras

# bleurt not supported out of the box with evaluate

pip install git+https://github.com/google-research/bleurt.git

pip install astunparse urllib  --force-reinstall --no-cache-dir

# openGL is no longer available on the cluster

OPENCV_VERSION=$(python - <<'EOF'
import importlib.metadata as m
try:
    print(m.version("opencv-python"))
except m.PackageNotFoundError:
    print(m.version("opencv-python-headless"))
EOF
)

pip install tensorflow==2.13.1 mediapipe==0.10.9 protobuf==3.20.3

pip uninstall -y opencv-python opencv-python-headless
pip install "opencv-python-headless==${OPENCV_VERSION}"

conda deactivate 
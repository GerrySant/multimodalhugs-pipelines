#! /bin/bash

############################
# mmposewholebody (py38)
############################
environment_scripts=$(dirname "$0")
scripts=$environment_scripts/../..
base=$scripts/..
venvs=$base/venvs

echo "location of mmposewholebody: $venvs/mmposewholebody"
conda activate $venvs/mmposewholebody
tools=$base/tools/mmposewholebody
mkdir -p $tools

# install fork of pose-format that extends to mmposewholebody

pip uninstall -y pose-format
git clone -b new_estimators https://github.com/catherine-o-brien/pose.git $tools/pose
cd $tools/pose/src/python
pip install -e .

# install dependencies for mmposewholebody
module load cuda/12.6.3 # required by mmcv-full
conda install pytorch torchvision 
pip install openmim 
mim install mmengine
mim install mmcv-full
mim install "mmdet>=3.1.0"
mim install mmpose

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

# openGL is no longer available on the cluster

OPENCV_VERSION=$(python - <<'EOF'
import importlib.metadata as m
try:
    print(m.version("opencv-python"))
except m.PackageNotFoundError:
    print(m.version("opencv-python-headless"))
EOF
)

pip uninstall -y opencv-python opencv-python-headless
pip install "opencv-python-headless==${OPENCV_VERSION}"
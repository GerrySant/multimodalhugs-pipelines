#! /bin/bash

############################
# openpifpaf (py310)
############################
base=$(dirname "$0")/
environment_scripts=$base/..
scripts=$environment_scripts/..
base=$scripts/..
venvs=$base/venvs

source activate $venvs/openpifpaf
tools=$base/tools/openpifpaf
mkdir -p $tools

# install fork of pose-format that extends to openpifpaf

pip uninstall -y pose-format
git clone -b new_estimators https://github.com/catherine-o-brien/pose.git $tools/pose
cd $tools/pose/src/python
pip install -e .

# install multimodalhugs

git clone https://github.com/GerrySant/multimodalhugs.git $tools/multimodalhugs

# pin commit  https://github.com/GerrySant/multimodalhugs/commit/5201c80f27aa70c460e8297a799dc5daccbd1b3b
# to avoid unintentionally breaking the code

(cd $tools/multimodalhugs && git checkout "5201c80f27aa70c460e8297a799dc5daccbd1b3b")

(cd $tools/multimodalhugs && pip install .)

# install openpifpaf
pip install "openpifpaf==0.13.11" --force-reinstall --no-cache-dir

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
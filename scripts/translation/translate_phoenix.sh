#! /bin/bash

# calling script needs to set:
# $base
# $dry_run
# $estimator
# $model_name

base=$1
dry_run=$2
estimator=$3
model_name=$4
env_name=${5}

if [[ -z "$env_name" ]]; then
    env_name="$estimator"
fi

venvs=$base/venvs
configs=$base/configs
configs_sub=$configs/$model_name/$estimator

models=$base/models
models_sub=$models/$model_name/$estimator

translations=$base/translations
translations_sub=$translations/$model_name/$estimator

mkdir -p $translations
mkdir -p $translations_sub

# measure time

SECONDS=0

################################

echo "Python before activating:"
which python

echo "activate path:"
which activate

echo "Executing: source activate $venvs/$env_name"
source activate $venvs/$env_name

echo "Python after activating:"
which python

# necessary?
# export CUDA_VISIBLE_DEVICES=0

# if in doubt, check with:
# echo "CUDA is available:"
# python -c 'import torch; print(torch.cuda.is_available())'

################################

# check if there are any checkpoints

model_name_or_path=$(ls -d "$models_sub"/train/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)

if [ -z "$model_name_or_path" ]; then
  echo "No checkpoints found in $models_sub"
  exit 1
fi

if [[ -s $translations_sub/generated_predictions.txt ]]; then
  echo "Translations exist: $translations_sub/generated_predictions.txt"
  echo "Skipping."
  exit 0
fi

if [[ $dry_run == "true" ]]; then
    use_cpu_arg="--use_cpu"
else
    use_cpu_arg=""
fi

multimodalhugs-generate \
    --task "translation" \
    --config_path $configs_sub/config-phoenix-$estimator.yaml \
    --metric_name "sacrebleu" \
    --generate_output_dir $translations_sub \
    --setup_path $models_sub/setup \
    --model_name_or_path $models_sub/train/checkpoint-best \
    --num_beams 5 $use_cpu_arg

echo "time taken:"
echo "$SECONDS seconds"

#! /bin/bash

# calling script needs to set:
# $base
# $dry_run
# $estimator
# $model_name
# $learning_rate
# $gradient_accumulation_steps
# $warmup_steps
# $batch_size
# $label_smoothing_factor
# $dataloader_num_workers
# $fp16
# $seed

base=$1
dry_run=$2
estimator=$3
model_name=$4
learning_rate=$5
gradient_accumulation_steps=$6
warmup_steps=$7
batch_size=$8
label_smoothing_factor=$9
dataloader_num_workers=${10}
fp16=${11}
seed=${12}
env_name=${13}
train_tsv_metadata_filename=${14}
validation_tsv_metadata_filename=${15}
test_tsv_metadata_filename=${16}


# if no environment name is passed, default to estimator
if [[ -z "$env_name" ]]; then
    env_name="$estimator"
fi

# if no train_tsv_metadata_filename name is passed, default to rwth_phoenix2014_t.train.tsv
if [[ -z "$train_tsv_metadata_filename" ]]; then
    train_tsv_metadata_filename="rwth_phoenix2014_t.train.tsv"
fi

# if no validation_tsv_metadata_filename name is passed, default to rwth_phoenix2014_t.validation.tsv
if [[ -z "$validation_tsv_metadata_filename" ]]; then
    validation_tsv_metadata_filename="rwth_phoenix2014_t.validation.tsv"
fi

# if no test_tsv_metadata_filename name is passed, default to rwth_phoenix2014_t.test.tsv
if [[ -z "$test_tsv_metadata_filename" ]]; then
    test_tsv_metadata_filename="rwth_phoenix2014_t.test.tsv"
fi

data=$base/data
preprocessed=$data/$estimator/preprocessed
scripts=$base/scripts
venvs=$base/venvs
configs=$base/configs
configs_with_model_name=$configs/$model_name
configs_sub=$configs_with_model_name/$estimator

models=$base/models
models_with_model_name=$models/$model_name
models_sub=$models_with_model_name/$estimator

mkdir -p $configs
mkdir -p $configs_with_model_name
mkdir -p $configs_sub
mkdir -p $models
mkdir -p $models_with_model_name
mkdir -p $models_sub

# skip if checkpoint exists

shopt -s nullglob
checkpoints=("$models_sub"/train/checkpoint*/)

if [ ${#checkpoints[@]} -gt 0 ]; then
    echo "Checkpoint folder exists, skipping"
    exit 0
else
    echo "No checkpoint folder, will start training"
fi

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

# setup

if [[ $dry_run == "true" ]]; then
    dry_run_arg="--dry-run"
    use_cpu_arg="--use_cpu"
    batch_size="1"
else
    dry_run_arg=""
    use_cpu_arg=""
fi

if [[ $fp16 == "true" ]]; then
    fp16_arg="--fp16"
else
    fp16_arg=""
fi

if [[ "$estimator" == "mediapipe" ]]; then
    reduce_holistic_poses_arg="--reduce-holistic-poses"
else
    reduce_holistic_poses_arg=""
fi

if [[ "$estimator" == "mediapipe" ]]; then
    feat_dim=534

elif [[ "$estimator" == "mmposewholebody" ]]; then
    feat_dim=266

elif [[ "$estimator" == "alphapose_133" ]]; then
    feat_dim=266

elif [[ "$estimator" == "alphapose_136" ]]; then
    feat_dim=272

elif [[ "$estimator" == "sapiens" ]]; then
    feat_dim=620

elif [[ "$estimator" == "smplest_x" ]]; then
    feat_dim=278

else
    echo "WARNING: Unknown estimator: $estimator. Defaulting to feat_dim=534"
    feat_dim=534
fi

python $scripts/training/create_config.py \
    --run-name "phoenix-$estimator" \
    --config-dir $configs_sub \
    --train-metadata-file $preprocessed/$train_tsv_metadata_filename \
    --validation-metadata-file $preprocessed/$validation_tsv_metadata_filename \
    --test-metadata-file $preprocessed/$test_tsv_metadata_filename \
    --new-vocabulary "__dgs__" \
    --feat-dim $feat_dim \
    --learning-rate $learning_rate \
    --gradient-accumulation-steps $gradient_accumulation_steps \
    --warmup-steps $warmup_steps \
    --batch-size $batch_size \
    --label-smoothing-factor $label_smoothing_factor \
    --dataloader-num-workers $dataloader_num_workers \
    --seed $seed \
    $reduce_holistic_poses_arg $dry_run_arg $fp16_arg

# https://github.com/GerrySant/multimodalhugs/issues/50

export HF_HUB_DISABLE_XET=1

# avoid writing to ~/.cache/huggingface

export HF_HOME=$data/$estimator/huggingface

multimodalhugs-setup \
    --modality "pose2text" \
    --config_path $configs_sub/config-phoenix-$estimator.yaml \
    --output_dir $models_sub \
    --seed $seed

# training

multimodalhugs-train \
    --task "translation" \
    --config_path $configs_sub/config-phoenix-$estimator.yaml \
    --setup_path $models_sub/setup \
    --output_dir $models_sub \
    --seed $seed \
    --report_to none $use_cpu_arg

echo "time taken:"
echo "$SECONDS seconds"

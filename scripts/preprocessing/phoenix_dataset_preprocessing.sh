#! /bin/bash

# calling script needs to set:
# $base
# $dry_run
# $estimator 

base=$1
dry_run=$2
estimator=$3
env_name=${4}

# if no environment name is passed, default to estimator
if [[ -z "$env_name" ]]; then
    env_name="$estimator"
fi


scripts=$base/scripts
data=$base/data
venvs=$base/venvs

estimator_data=$data/$estimator
poses=$estimator_data/poses
preprocessed=$estimator_data/preprocessed

echo "preprocessed: $preprocessed"

mkdir -p $data
mkdir -p $estimator_data
mkdir -p $poses $preprocessed

shopt -s nullglob nocaseglob

train_files=("$preprocessed"/*train*.tsv)

found=${#train_files[@]}

shopt -u nocaseglob

if [ "$found" -gt 0 ]; then
    echo "Preprocessed train TSV file(s) found:"
    printf '  %s\n' "${train_files[@]}"
    exit 0
else
    echo "No preprocessed train TSV files found yet"
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

################################

if [[ $dry_run == "true" ]]; then
    dry_run_arg="--dry-run"
else
    dry_run_arg=""
fi

python $scripts/preprocessing/phoenix_dataset_preprocessing.py \
    --estimator $estimator \
    --pose-dir $poses \
    --output-dir $preprocessed \
    --tfds-data-dir $data/tensorflow_datasets \
    --video-dir $data/phoenix_videos $dry_run_arg

# sizes
echo "Sizes of preprocessed TSV files:"

wc -l $preprocessed/*

echo "time taken:"
echo "$SECONDS seconds"
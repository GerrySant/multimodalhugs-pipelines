#! /bin/bash

running_scripts=$(dirname "$0")
base=$running_scripts/../..
base=$(realpath $base)
scripts=$base/scripts

# set to "false" or "true":

dry_run="false"

# set to your desired estimator
estimator="smplest_x" # options: mmposewholebody, mediapipe, openpifpaf, smplest_x, sapiens, alphapose_133, alphapose_136

model_name="phoenix_$estimator"

# best hyperparams found so far

learning_rate="1e-5"
warmup_steps=500
label_smoothing_factor="0.1"
gradient_accumulation_steps=3


if [[ "$estimator" == "smplest_x" \
   || "$estimator" == "sapiens" \
   || "$estimator" == "alphapose_133" \
   || "$estimator" == "alphapose_136" ]]; then

    train_tsv_metadata_filename="PHOENIX-2014-T.train.corpus_${estimator}_poses.tsv"
    validation_tsv_metadata_filename="PHOENIX-2014-T.validation.corpus_${estimator}_poses.tsv"
    test_tsv_metadata_filename="PHOENIX-2014-T.test.corpus_${estimator}_poses.tsv"
    env_name="multiple_support"
fi

. $scripts/running/run_generic.sh
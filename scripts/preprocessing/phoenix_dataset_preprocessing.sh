#! /bin/bash

# calling script needs to set:
# $base
# $dry_run
# $estimator 
# $env_name 

# base=$1
# dry_run=$2
# estimator=$3
# env_name=${4}

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

mkdir -p $data
mkdir -p $estimator_data
mkdir -p $poses $preprocessed

shopt -s nullglob nocaseglob
train_files=("$preprocessed"/*train*.tsv)
found=${#train_files[@]}
shopt -u nullglob nocaseglob

if [ "$found" -gt 0 ]; then
    echo "Preprocessed train TSV file(s) found:"
    printf '  %s\n' "${train_files[@]}"
    exit 0
else
    echo "No preprocessed train TSV files found yet"
fi

# measure time
SECONDS=0

if [[ $dry_run == "true" ]]; then
    dry_run_arg="--dry-run"
else
    dry_run_arg=""
fi

################################
# estimator-specific execution
################################

case "$estimator" in

    smplest_x)
        echo "Activating environment for SMPLest-X"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $venvs/smplest_x_pose_extractor

        echo "Running SMPLest-X pose extraction"

        # save current directory
        current_dir="$(pwd)"
        repo_dir="$base/tools/smplest-x"
        cd "$repo_dir" || { echo "Cannot cd to $repo_dir"; exit 1; }

        export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}

        json_poses=$estimator_data/json_poses

        # create estimator structure
        mkdir -p "$json_poses"
        mkdir -p "$poses"

        for split in train validation test; do
            videos_path="$data/phoenix_videos/$split"
            output_path="$json_poses/$split"

            mkdir -p "$output_path"

            echo "Processing split: $split"
            python main/json_pose_estimator.py \
                --video_path "$videos_path" \
                --ckpt_name "smplest_x_h" \
                --json_output_path "$output_path"
        done

        # return to original directory
        cd "$current_dir" || exit 1

        conda deactivate

        # Convert JSON → Pose format
        echo "Activating environment for pose conversion"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $venvs/$env_name

        for split in train validation test; do
            videos_path="$data/phoenix_videos/$split"
            json_split="$json_poses/$split"
            pose_split="$poses/$split"

            mkdir -p "$pose_split"

            for json_file in "$json_split"/*.json; do
                filename="$(basename "$json_file" .json)"

                video_file="$videos_path/$filename.mp4"
                pose_file="$pose_split/$filename.pose"

                json_to_pose \
                    -i "$json_file" \
                    -o "$pose_file" \
                    --original-video "$video_file" \
                    --format smplest-x
            done
        done

        conda deactivate

        # Create the tsv files
        echo "Re-activating default environment for preprocessing"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $venvs/$env_name

        python $scripts/preprocessing/phoenix_dataset_preprocessing.py \
            --estimator $estimator \
            --pose-dir $poses \
            --output-dir $preprocessed \
            --tfds-data-dir $data/tensorflow_datasets \
            --video-dir $data/phoenix_videos \
            --poses-precomputed \
            $dry_run_arg
        ;;

    sapiens)
        echo "Activating environment for Sapiens"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $venvs/sapiens_pose_extractor

        echo "Running Sapiens pose extraction"

        export PYOPENGL_PLATFORM=osmesa
        json_poses=$estimator_data/json_poses

        # create estimator structure
        mkdir -p "$json_poses"
        mkdir -p "$poses"

        for split in train validation test; do
            videos_path="$data/phoenix_videos/$split"
            output_path="$json_poses/$split"

            mkdir -p "$output_path"

            echo "Processing split: $split"
            python $scripts/preprocessing/sapiens_estimation.py \
                --video_path "${videos_path}" \
                --output_path "${output_path}" \
                --model_type "POSE_ESTIMATION_1B"
        done

        conda deactivate

        # Convert JSON → Pose format
        echo "Activating environment for pose conversion"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $venvs/$env_name

        for split in train validation test; do
            videos_path="$data/phoenix_videos/$split"
            json_split="$json_poses/$split"
            pose_split="$poses/$split"

            mkdir -p "$pose_split"

            for json_file in "$json_split"/*.json; do
                filename="$(basename "$json_file" .json)"

                video_file="$videos_path/$filename.mp4"
                pose_file="$pose_split/$filename.pose"

                json_to_pose \
                    -i "$json_file" \
                    -o "$pose_file" \
                    --original-video "$video_file" \
                    --format sapiens
            done
        done

        conda deactivate

        # Create the tsv files
        echo "Re-activating default environment for preprocessing"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $venvs/$env_name

        python $scripts/preprocessing/phoenix_dataset_preprocessing.py \
            --estimator $estimator \
            --pose-dir $poses \
            --output-dir $preprocessed \
            --tfds-data-dir $data/tensorflow_datasets \
            --video-dir $data/phoenix_videos \
            --poses-precomputed \
            $dry_run_arg
        ;;

    alphapose_133|alphapose_136)
        echo "Activating environment for AlphaPose"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $venvs/alphapose_pose_extractor

        echo "Running AlphaPose pose extraction"

        # save current directory
        current_dir="$(pwd)"
        repo_dir="$base/tools/alphapose/AlphaPose"
        cd "$repo_dir" || { echo "Cannot cd to $repo_dir"; exit 1; }

        export PYOPENGL_PLATFORM=osmesa

        json_poses=$estimator_data/json_poses

        if [[ "$estimator" == "alphapose_133" ]]; then
        checkpoint="${repo_dir}/pretrained_models/coco_wholebody133_fast50_regression_256x192.pth"
        config="${repo_dir}/configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-regression.yaml"
        else
        # we assume $estimator = "alphapose_136"
        checkpoint="${repo_dir}/pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth"
        config="${repo_dir}/configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml"
        fi

        # create estimator structure
        mkdir -p "$json_poses"
        mkdir -p "$poses"

        for split in train validation test; do
            videos_path="$data/phoenix_videos/$split"
            output_path="$json_poses/$split"

            mkdir -p "$output_path"

            echo "Processing split: $split"

            python $scripts/preprocessing/alphapose_estimation.py \
                --cfg $config \
                --checkpoint $checkpoint \
                --video $videos_path \
                --outdir $output_path \
                --sp
        done

        # return to original directory
        cd "$current_dir" || exit 1

        conda deactivate

        # Convert JSON → Pose format
        echo "Activating environment for pose conversion"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $venvs/$env_name

        for split in train validation test; do
            videos_path="$data/phoenix_videos/$split"
            json_split="$json_poses/$split"
            pose_split="$poses/$split"

            mkdir -p "$pose_split"

            for json_file in "$json_split"/*.json; do
                filename="$(basename "$json_file" .json)"

                video_file="$videos_path/$filename.mp4"
                pose_file="$pose_split/$filename.pose"

                json_to_pose \
                    -i "$json_file" \
                    -o "$pose_file" \
                    --original-video "$video_file" \
                    --format alphapose
            done
        done

        conda deactivate

        # Create the tsv files
        echo "Re-activating default environment for preprocessing"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $venvs/$env_name

        python $scripts/preprocessing/phoenix_dataset_preprocessing.py \
            --estimator $estimator \
            --pose-dir $poses \
            --output-dir $preprocessed \
            --tfds-data-dir $data/tensorflow_datasets \
            --video-dir $data/phoenix_videos \
            --poses-precomputed \
            $dry_run_arg
        ;;

    *)
        echo "Activating default environment"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $venvs/$env_name

        echo "Running default preprocessing"

        python $scripts/preprocessing/phoenix_dataset_preprocessing.py \
            --estimator $estimator \
            --pose-dir $poses \
            --output-dir $preprocessed \
            --tfds-data-dir $data/tensorflow_datasets \
            --video-dir $data/phoenix_videos \
            $dry_run_arg
        ;;
esac

# sizes
echo "Sizes of preprocessed TSV files:"

wc -l $preprocessed/*

echo "time taken:"
echo "$SECONDS seconds"
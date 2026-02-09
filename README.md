# multimodalhugs-pipelines

Download the code:

    git clone https://github.com/bricksdont/multimodalhugs-pipelines
    cd multimodalhugs-pipelines

## Basic setup

Create venvs for each estimator:

    ./scripts/environment/create_envs.sh

Then install required software to each venv:

    ./scripts/environment/install.sh


## Notes on data

Due to the deprecation of `SignDataConfig`, the automatic loading of the Phoenix dataset is currently broken. 

If you have access to /shares/iict-sp2.ebling.cl.uzh/ on the UZH Science-IT cluster, you can download the videos with this script (approx. 50 GB of data):

    cd multimodalhugs-pipelines
    ./scripts/data-loading/load-data-uzh.sh

If you are not a member of UZH Science IT, you must yourself put the Phoenix dataset .mp4 files in these three folders: 

    /multimodalhugs-pipelines/data/phoenix_videos/validation
    /multimodalhugs-pipelines/data/phoenix_videos/train
    /multimodalhugs-pipelines/data/phoenix_videos/test

The three metadata .tsv files must be located at:

    /multimodalhugs-pipelines/data/phoenix_videos/PHOENIX-2014-T.validation.corpus_poses.tsv
    /multimodalhugs-pipelines/data/phoenix_videos/PHOENIX-2014-T.train.corpus_poses.tsv
    /multimodalhugs-pipelines/data/phoenix_videos/PHOENIX-2014-T.test.corpus_poses.tsv

Given that the Phoenix dataset is only publicly available in image files, not .mp4, this is a massive burden to the user. Adding a script to download the data and generate the .mp4 files for non-Science IT users is planned for the future. 

## Run experiments

### Single experiment

It is a good idea to use `dry_run="true"` the first time you are running code, which creates all files, executes all
the code but uses only a fraction of the training data, trains for very few steps only, etc - as a
general sanity check. If you want to launch a real run after a dry run you will need to manually
delete folders that the dry run created (e.g. a sub-folder of `models`), otherwise the steps
will not be repeated.

Then to train a basic model:

    ./scripts/running/run_basic.sh

Automatically, pose estimation is set to use `mediapipe`. If you would like to use a different estimator, modify the `estimator` variable in `run_basic.sh`. 

This will train a basic MultimodalHugs model. All steps are submitted
as SLURM jobs.

If the process is fully reproducible, this should result in a test set BLEU score of `10.691`. This
value is inside the file `evaluations/phoenix/test_score.bleu`.

### Hyperparams exploration

The following script will train approximately 50 models to search for good hyperparameters
(each run will finish in roughly 2 hours):

    ./scripts/running/run_hyperparam_search.sh

To get a summary of the results, run:

    ./scripts/summaries/summarize.sh

### Check if training / generation is reproducible

    ./scripts/running/run_test_repeatability.sh

This will train three models with identical configurations and seeds, to test if the process is repeatable / reproducible.

**Results**

|           | test BLEU | stopped training at epoch |
|-----------|-----------|---------------------------|
| phoenix_1 | 10.199    | 29.5051                   |
| phoenix_2 | 10.217    | 22.5627                   |
| phoenix_3 | 10.472    | 26.0339                   |

**Investigating if due to training arguments**

Using only a single data worker:

|                     | test BLEU | stopped training at epoch |
|---------------------|------|---------------------------|
| phoenix_1_workers_1 | 10.324  | 23.4305                   |
| phoenix_2_workers_1 | 10.244 | 32.1085                   |
| phoenix_3_workers_1 | 10.189 | 21.261                   |

fp32 instead of fp16:

|                | test BLEU | stopped training at epoch |
|----------------|--------|--------------------------|
| phoenix_1_fp32 | 10.35  | 29.5051                  |
| phoenix_2_fp32      | 10.5  | 22.5627                         |
| phoenix_3_fp32      | 10.108  | 27.3356                         |

fp32 and a single data worker:

|           | test BLEU | stopped training at epoch |
|-----------|------|-----------------------|
| phoenix_1_fp32_workers_1 | 10.379 | 27.7695                  |
| phoenix_2_fp32_workers_1 | 9.546  | 21.261                      |
| phoenix_3_fp32_workers_1 | 10.111 | 25.6                   |

**Investigate if due to weight initialization**

Investigating whether the model weights at checkpoint zero (the setup model) are identical for two models, here
are the ones that are not identical:

| key       | magnitude of difference |
|-----------|-------------------|
| multimodal_mapper.mapping_layer.weight | 0.08649232983589172            |
| multimodal_mapper.mapping_layer.bias | 0.08635331690311432            |
| backbone.model.shared.weight | 0.0894961804151535            |
| backbone.model.encoder.embed_tokens.weight | 0.0894961804151535            |
| backbone.model.decoder.embed_tokens.weight | 0.0894961804151535               |
| backbone.lm_head.weight | 0.0894961804151535               |

These results are generated by [scripts/debugging/debug_reproducibility.py](https://github.com/bricksdont/multimodalhugs-examples/blob/main/scripts/debugging/debug_reproducibility.py).

The magnitude of differences seems to indicate that the weight initialization is different, potentially because the seed is not set
during setup, when the multimodal mapper weights are created. Specifically, the [build_model method of the
MultimodalEmbedderModel](https://github.com/GerrySant/multimodalhugs/blob/master/multimodalhugs/models/multimodal_embedder/modeling_multimodal_embedder.py#L258) runs without a fixed seed.

**After fixing the random seed before setup**

After applying [this fix](https://github.com/GerrySant/multimodalhugs/compare/master...bricksdont:multimodalhugs:fix_reproducibility):

|           | test BLEU | stopped training at epoch |
|-----------|-----|------------------|
| phoenix_1 | 9.982 | 24.2983             |
| phoenix_2 | 9.982 | 24.2983             |
| phoenix_3 | 9.982 | 24.2983             |

also, all initial weight parameters are identical between models.

# Adding a new pose estimator
To add a new pose estimator:
1. Merge the implementation of your new estimator into the new-estimators branch of `https://github.com/catherine-o-brien/pose.git`
2. Add a command to create a conda environment for your estimator in the file `/scripts/environment/install.sh`. 

    *Note: the name you give your estimator MUST be the same as its estimator name in the `pose` repo.*
3. Add an installation script to `/scripts/environment/install-scripts.sh`
4. After that, set the `estimator` variable in `scripts/running/run_basic.sh` to your estimator, run the repo as normal, making sure to rerun the `./create_envs.sh` and `./install-all.sh` shell scripts. 

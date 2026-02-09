#!/bin/bash
#SBATCH --job-name=copy_phoenix
#SBATCH --output=copy_phoenix_%j.out
#SBATCH --error=copy_phoenix_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

#######################################
# Base directory (where this script lives)
#######################################
base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export base_dir

echo "Base directory: $base_dir"

#######################################
# Source and destination roots
#######################################
SRC_VIDEO_BASE="/shares/iict-sp2.ebling.cl.uzh/cobrie/multimodalhugs-pipelines/data/phoenix_videos"
SRC_TSV_BASE="/shares/iict-sp2.ebling.cl.uzh/cobrie/multimodalhugs-pipelines/data/phoenix_videos"

DST_BASE="$base_dir/data/phoenix_videos"

#######################################
# 1. Copy video directories (skip if non-empty)
#######################################
for split in validation test train; do
    src_dir="$SRC_VIDEO_BASE/$split/videos"
    dst_dir="$DST_BASE/$split"

    if [[ -d "$dst_dir" && "$(ls -A "$dst_dir" 2>/dev/null)" ]]; then
        echo "Skipping $split videos (already exists and non-empty)"
    else
        echo "Copying $split videos with rsync"
        mkdir -p "$dst_dir"
        rsync -avh --progress2 "$src_dir/" "$dst_dir/"
    fi
done

#######################################
# 2. Copy + rewrite TSV files (skip if exists)
#######################################
mkdir -p "$DST_BASE"

copy_and_rewrite_tsv() {
    local split="$1"
    local filename="PHOENIX-2014-T.${split}.corpus_poses.tsv"
    local src_tsv="$SRC_TSV_BASE/$filename"
    local dst_tsv="$DST_BASE/$filename"

    if [[ -f "$dst_tsv" ]]; then
        echo "Skipping $filename (already exists)"
        return
    fi

    echo "Copying $filename"
    rsync -avh "$src_tsv" "$dst_tsv"

    echo "Rewriting video_path in $filename"

    awk -F'\t' -v OFS='\t' -v base="$DST_BASE" -v split="$split" '
    NR==1 {
        for (i = 1; i <= NF; i++) {
            if ($i == "video_path") {
                vp_col = i
            }
        }
        if (!vp_col) {
            print "ERROR: video_path column not found" > "/dev/stderr"
            exit 1
        }
        print
        next
    }
    {
        video_name = $vp_col
        sub(/^.*\//, "", video_name)
        $vp_col = base "/" split "/" video_name
        print
    }
    ' "$dst_tsv" > "${dst_tsv}.tmp"

    mv "${dst_tsv}.tmp" "$dst_tsv"
}

copy_and_rewrite_tsv dev
copy_and_rewrite_tsv test
copy_and_rewrite_tsv train

echo "PHOENIX-2014-T copy completed successfully."

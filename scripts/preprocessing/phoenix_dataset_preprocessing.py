import os
import glob
import subprocess
import argparse
import logging
import csv
import time

from typing import Iterator, Dict, List, Optional

Example = Dict[str, str]

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and transform a TSV file.")
    parser.add_argument("--estimator", type=str, help="Which pose estimator to use.")
    parser.add_argument("--pose-dir", type=str, help="Where to save poses.")
    parser.add_argument("--output-dir", type=str, help="Path to the output TSV files.")
    parser.add_argument("--encoder-prompt", type=str, default="__dgs__", help="encoder prompt string.")
    parser.add_argument("--decoder-prompt", type=str, default="__de__", help="decoder prompt string.")
    parser.add_argument("--tfds-data-dir", type=str, default=None,
                        help="TFDS data folder to cache downloads.", required=False)
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Location of the Phoenix dataset videos.", required=False)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Process very few elements only.", required=False)
    return parser.parse_args()

def load_phoenix_videos():
    #TODO: implement logic that downloads the Phoenix images and turns them into videos and saves that into the data directory. 
    #For now, assume that the data is already present at --video-dir in three partitions (validation, train, test)
    pass

def load_split_metadata(video_dir: str, split_name: str) -> Dict[str, str]:
    """
    Loads the Phoenix TSV metadata for a given split and returns a mapping
    from video basename (without extension) -> German text.
    """
    tsv_path = os.path.join(
        video_dir,
        f"PHOENIX-2014-T.{split_name}.corpus_poses.tsv"
    )

    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Metadata TSV not found: {tsv_path}")

    mapping = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Extract basename of video file (without extension)
            video_filename = os.path.basename(row["signal"])
            datum_id = os.path.splitext(video_filename)[0]
            mapping[datum_id] = row["output"]
    return mapping

def generate_examples(estimator: str,
                      video_dir: str,
                      split_name: str,
                      split_pose_dir: str,
                      dry_run: bool = False) -> Iterator[Example]:
    """
    :param video_dir: The estimator to use: select from: "mmposewholebody"
    :param video_dir: Base path, e.g. "{base}/data/phoenix_videos"
    :param split_name: "train", "validation", or "test" (these should be directories inside video_dir)
    :param pose_dir: Directory where .pose files will be written
    :param dry_run: If True, process only a small number of videos
    """

    split_dir = os.path.join(video_dir, split_name)
    if not os.path.isdir(split_dir):
        raise ValueError(f"Split directory does not exist: {split_dir}")

    os.makedirs(split_pose_dir, exist_ok=True)

    video_paths = sorted(glob.glob(os.path.join(split_dir, "*.mp4")))

    if dry_run:
        video_paths = video_paths[:10]

    logging.info(f"Found {len(video_paths)} videos in {split_dir}")

    # Load TSV metadata mapping
    datum_text_mapping = load_split_metadata(video_dir, split_name)

    for video_path in video_paths:
        tic = time.perf_counter()
        datum_id = os.path.splitext(os.path.basename(video_path))[0]
        pose_filepath = os.path.join(split_pose_dir, f"{datum_id}.pose")

        # skip if already processed
        if os.path.exists(pose_filepath):
            logging.debug(f"Pose already exists, skipping: {pose_filepath}")
        
        else:
            cmd = [
                "video_to_pose", 
                "--format", estimator, 
                "-i", video_path, 
                "-o", pose_filepath,
            ] #TODO: change so that if dry-run is true, we use cpu processing 
            
            if dry_run:
                logging.debug("Running command: %s", " ".join(cmd))

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                logging.error("Pose estimation failed for %s", video_path)
                logging.error("STDOUT:\n%s", e.stdout)
                logging.error("STDERR:\n%s", e.stderr)
                raise
        
        text = datum_text_mapping.get(datum_id, "")
        if not text:
            logging.warning(f"No German text found for video: {datum_id}")

        yield {
            "datum_id": datum_id, 
            "video_filepath": video_path, 
            "text": text,
            "pose_filepath": pose_filepath
        }
        logging.debug(f"Processed video {datum_id} in {time.perf_counter() - tic:.2f} seconds\n")

def write_examples_tsv(examples: List[Example],
                       output_dir: str,
                       encoder_prompt: str,
                       decoder_prompt: str,
                       split_name: str,):
    """
    If signal_start and signal_end are not required (when all the frames are used), must be set as 0.

    :param examples:
    :param output_dir:
    :param encoder_prompt:
    :param decoder_prompt:
    :param split_name:
    :return:
    """

    filepath = os.path.join(output_dir, f"rwth_phoenix2014_t.{split_name}.tsv")

    logging.debug("Writing generated examples to: '%s'" % filepath)

    fieldnames = ["signal", "signal_start", "signal_end", "encoder_prompt", "decoder_prompt", "output"]

    with open(filepath, "w", newline="") as outhandle:
        writer = csv.DictWriter(outhandle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for example in examples:
            row_dict = {
                "signal": example["pose_filepath"],
                "signal_start": 0,
                "signal_end": 0,
                "encoder_prompt": encoder_prompt,
                "decoder_prompt": decoder_prompt,
                "output": example["text"]
            }

            writer.writerow(row_dict)

def main():
    # Parse arguments
    args = parse_arguments()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    logging.debug(f"Doing pose estimation with estimator: {args.estimator}")

    load_phoenix_videos()

    stats = {}

    for split_name in ["train", "test", "validation"]:

        split_pose_dir = os.path.join(args.pose_dir, split_name)
        os.makedirs(split_pose_dir, exist_ok=True)

        logging.debug(f"\n\n\nGenerating examples for split: {split_name}\n")
        
        examples = list(generate_examples(estimator=args.estimator,
                                            video_dir=args.video_dir,
                                            split_name=split_name,
                                            split_pose_dir=split_pose_dir,
                                            dry_run=args.dry_run))
        
        print(examples)

        stats[split_name] = len(examples)

        write_examples_tsv(examples=examples,
                    output_dir=args.output_dir,
                    encoder_prompt=args.encoder_prompt,
                    decoder_prompt=args.decoder_prompt,
                    split_name=split_name)

    logging.debug("Number of examples found:")
    logging.debug(stats)



if __name__ == "__main__":
    main()
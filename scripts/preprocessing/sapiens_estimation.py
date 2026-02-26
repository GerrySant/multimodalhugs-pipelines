import argparse
import os
import json
from tqdm import tqdm

import torch
import cv2

import sapiens_inference
from sapiens_inference.pose import (
    SapiensPoseEstimation,
    SapiensPoseEstimationType
)
from sapiens_inference.pose_classes_and_palettes import (
    GOLIATH_SKELETON_INFO,
    GOLIATH_KEYPOINTS
)

def ensure_sapiens_cwd():
    """
    Ensures the working directory is the root of the
    Sapiens-Pytorch-Inference repository.
    """
    pkg_path = os.path.dirname(os.path.abspath(sapiens_inference.__file__))
    repo_root = os.path.dirname(pkg_path)

    current_cwd = os.getcwd()
    if current_cwd != repo_root:
        print(f"🔄 Changing working directory to: {repo_root}")
        os.chdir(repo_root)

# ---------------------------------
# Device
# ---------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    else:
        return torch.device("cpu"), torch.float32


# ---------------------------------
# Process a single video
# ---------------------------------
def process_video(
    video_path,
    output_json_path,
    estimator,
    max_frames=None
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_frames = min(num_frames, max_frames) if max_frames else num_frames

    output_data = {
        "metadata": {
            "video": os.path.basename(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "num_keypoints": len(GOLIATH_KEYPOINTS),
            "keypoint_names": GOLIATH_KEYPOINTS,
            "skeleton": [
                link["link"] for link in GOLIATH_SKELETON_INFO.values()
            ],
        },
        "frames": []
    }

    for frame_idx in tqdm(
        range(total_frames),
        desc=f"Processing {os.path.basename(video_path)}",
        leave=False
    ):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        with torch.no_grad():
            bboxes = estimator.detector.detect(frame)
            if len(bboxes) == 0:
                continue

            bbox = bboxes[0]
            _, all_keypoints = estimator.estimate_pose(frame, [bbox])
            kpts = all_keypoints[0]

        frame_keypoints = {
            name: [float(x), float(y), float(score)]
            for name, (x, y, score) in kpts.items()
        }

        output_data["frames"].append({
            "frame": frame_idx,
            "timestamp": timestamp,
            "box": [float(v) for v in bbox[:4]],
            "keypoints": frame_keypoints
        })

    cap.release()

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Saved: {output_json_path}")


# ---------------------------------
# Main
# ---------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract SAPIENS pose keypoints from video(s)"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to a .mp4 file or a directory containing .mp4 files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory (or output .json if single video)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process per video"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="POSE_ESTIMATION_1B",
        help="Sapiens model type (e.g. POSE_ESTIMATION_1B)"
    )

    args = parser.parse_args()

    ensure_sapiens_cwd()
    print("Current working directory:", os.getcwd())

    model_type = getattr(SapiensPoseEstimationType, args.model_type)

    device, dtype = get_device()

    estimator = SapiensPoseEstimation(
        type=model_type,
        device=device,
        dtype=dtype,
    )

    if os.path.isfile(args.video_path):
        # Single video
        if args.output_path.endswith(".json"):
            output_json = args.output_path
        else:
            os.makedirs(args.output_path, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.video_path))[0]
            output_json = os.path.join(args.output_path, base + ".json")

        process_video(
            args.video_path,
            output_json,
            estimator,
            args.max_frames
        )

    elif os.path.isdir(args.video_path):
        # Directory of videos
        os.makedirs(args.output_path, exist_ok=True)

        videos = sorted(
            f for f in os.listdir(args.video_path)
            if f.lower().endswith(".mp4")
        )

        for video in videos:
            video_path = os.path.join(args.video_path, video)
            base = os.path.splitext(video)[0]
            output_json = os.path.join(args.output_path, base + ".json")

            process_video(
                video_path,
                output_json,
                estimator,
                args.max_frames
            )

    else:
        raise ValueError("❌ video_path is neither a file nor a directory")


if __name__ == "__main__":
    main()

"""
Qualitative analysis: sweep missing-hand percentage across pose estimators.

Expected directory structures
------------------------------

When using ``--base-path``, the base directory must contain one sub-directory
per pose estimator, each holding either:

  a) A single .pose file:

      <base-path>/
      ├── mediapipe/
      │   └── poses.pose
      ├── alphapose_133/
      │   └── poses.pose
      └── sapiens/
          └── poses.pose

  b) Multiple .pose files (one per video / sentence), which are loaded and
     concatenated in sorted order along the time axis:

      <base-path>/
      ├── mediapipe/
      │   ├── sentence_001.pose
      │   ├── sentence_002.pose
      │   └── sentence_003.pose
      ├── alphapose_133/
      │   ├── sentence_001.pose
      │   └── ...
      └── ...

When using ``--path``, the same rules apply but the single path is shared
across all estimators (useful when comparing a single estimator's output or
when all estimators write into a common directory).

Usage examples
--------------

  # Sweep over estimators using a base directory (path/{estimator}/)
  python missing_hand_analysis.py --base-path /path/to/qualitative_evaluation/media/signsuisse

  # Single path, all estimators share the same file/directory
  python missing_hand_analysis.py --path /path/to/poses/

  # Restrict to specific estimators
  python missing_hand_analysis.py --base-path /path/to/... --estimators mediapipe alphapose_133

  # Adjust thresholds
  python missing_hand_analysis.py --path /path/to/poses/ \\
      --confidence-threshold 1e-12 \\
      --missing-thresholds 10 25 50 75 100
"""

import argparse
import os

import numpy as np
from pose_format import Pose
from pose_format.utils.generic import (
    detect_known_pose_format,
    get_body_hand_wrist_index,
    get_component_names,
)

# --------------------------------------------------
# File cache
# --------------------------------------------------

_last_buffer = None
_last_file_path = None


def _read_pose(file_path: str) -> bytes:
    global _last_buffer, _last_file_path
    if _last_file_path != file_path:
        with open(file_path, "rb") as f:
            _last_buffer = f.read()
            _last_file_path = file_path
    return _last_buffer


# --------------------------------------------------
# Load single or multiple poses
# --------------------------------------------------

def load_pose_or_directory(path: str) -> Pose:
    """
    If path is file → load pose.
    If path is directory → load all .pose files and concatenate frames (axis=0).
    """
    if os.path.isfile(path):
        buffer = _read_pose(path)
        pose = Pose.read(buffer)
        pose.body.data = pose.body.data[:, :1]
        pose.body.confidence = pose.body.confidence[:, :1]
        return pose

    if os.path.isdir(path):
        pose_files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".pose")
        )

        if not pose_files:
            raise ValueError(f"Directory contains no .pose files: {path}")

        poses = []
        for p in pose_files:
            buffer = _read_pose(p)
            pose = Pose.read(buffer)
            pose.body.data = pose.body.data[:, :1]
            pose.body.confidence = pose.body.confidence[:, :1]
            poses.append(pose)

        header = poses[0].header
        for p in poses[1:]:
            if p.header.total_points() != header.total_points():
                raise ValueError(
                    "All poses must share the same schema to concatenate. "
                    f"Got {header.total_points()} vs {p.header.total_points()} in {path}"
                )

        data = np.concatenate([p.body.data for p in poses], axis=0)
        conf = np.concatenate([p.body.confidence for p in poses], axis=0)
        poses[0].body.data = data
        poses[0].body.confidence = conf
        return poses[0]

    raise ValueError(f"Invalid path: {path}")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_only_one_hand_pose(pose: Pose, side: str) -> Pose:
    """
    Return a copy of `pose` that keeps only the components belonging to the
    requested hand, matched by component names containing both the side
    ("left"/"right") and "hand".
    """
    side_pose = pose.copy()
    component_names = get_component_names(side_pose)
    to_remove = [
        name for name in component_names
        if not (side.lower() in name.lower() and "hand" in name.lower())
    ]
    return side_pose.remove_components(components_to_remove=to_remove)


def _get_elbow_index(pose: Pose, hand: str) -> int:
    """Return elbow index for any supported pose format. `hand` must be 'LEFT' or 'RIGHT'."""
    fmt = detect_known_pose_format(pose)

    if fmt == "holistic":
        return pose.header.get_point_index("POSE_LANDMARKS", f"{hand}_ELBOW")
    if fmt in ["openpose", "openpose_135"]:
        comp = "pose_keypoints_2d" if fmt == "openpose" else "BODY_135"
        return pose.header.get_point_index(comp, f"{hand[0]}Elbow")
    if fmt == "smplest-x":
        return pose.header.get_point_index("BODY", f"{hand[0]}_Elbow")
    if fmt in ["alphapose_133", "alphapose_136"]:
        return pose.header.get_point_index(f"BODY_{fmt[-3:]}", f"{hand.lower()}_elbow")
    if fmt == "sapiens":
        return pose.header.get_point_index("BODY_SAPIENS", f"{hand.lower()}_olecranon")
    if fmt == "coco_wholebody_133":
        return pose.header.get_point_index("BODY", f"{hand.lower()}_elbow")

    raise NotImplementedError(fmt)


# --------------------------------------------------
# Metric
# --------------------------------------------------

def compute_signing_missing_hand_percentage(
    pose: Pose,
    hand: str = "BOTH",
    confidence_threshold: float = 0.2,
    missing_ratio_threshold: float = 0.6,
    verbose: bool = False,
) -> float:
    """
    Percentage of "signing frames" where a hand is considered missing.

    - signing frames: wrist_y < elbow_y (per-hand, OR for BOTH)
    - a hand is missing in a frame if:
        (# hand keypoints with conf < confidence_threshold) / (# hand keypoints)
        > missing_ratio_threshold

    Returns a percentage in [0, 100].
    """
    hand = hand.upper()
    assert hand in ["LEFT", "RIGHT", "BOTH"]

    def wrist_idx(h):
        return get_body_hand_wrist_index(pose, h)

    def elbow_idx(h):
        return _get_elbow_index(pose, h)

    def signing_mask(w_idx, e_idx):
        wrist_y = pose.body.data[:, 0, w_idx, 1]
        elbow_y = pose.body.data[:, 0, e_idx, 1]
        return wrist_y < elbow_y

    def hand_missing_ratio(side):
        hand_pose = get_only_one_hand_pose(pose, side)
        conf = hand_pose.body.confidence[:, 0, :]  # (T, K_hand)
        missing_points = conf < confidence_threshold
        return missing_points.sum(axis=1) / conf.shape[1]

    if hand in ["LEFT", "RIGHT"]:
        w, e = wrist_idx(hand), elbow_idx(hand)
        is_signing = signing_mask(w, e)
        missing = hand_missing_ratio(hand) > missing_ratio_threshold
        missing_during_signing = np.logical_and(is_signing, missing)

        if verbose:
            print(f"[{hand}] signing frames:", np.where(is_signing)[0])
            print(f"[{hand}] missing frames:", np.where(missing_during_signing)[0])

        if is_signing.sum() == 0:
            return 0.0
        return 100.0 * missing_during_signing.sum() / is_signing.sum()

    # BOTH
    lw, le = wrist_idx("LEFT"), elbow_idx("LEFT")
    rw, re = wrist_idx("RIGHT"), elbow_idx("RIGHT")
    signing = np.logical_or(signing_mask(lw, le), signing_mask(rw, re))

    left_missing = hand_missing_ratio("LEFT") > missing_ratio_threshold
    right_missing = hand_missing_ratio("RIGHT") > missing_ratio_threshold
    both_missing = np.logical_and(left_missing, right_missing)
    missing_during_signing = np.logical_and(signing, both_missing)

    if verbose:
        print("[BOTH] signing frames:", np.where(signing)[0])
        print("[BOTH] missing frames:", np.where(missing_during_signing)[0])

    if signing.sum() == 0:
        return 0.0
    return 100.0 * missing_during_signing.sum() / signing.sum()


# --------------------------------------------------
# CLI
# --------------------------------------------------

DEFAULT_ESTIMATORS = [
    "mediapipe",
    "alphapose_133",
    "alphapose_136",
    "sapiens",
    "smplest_x",
    "mmposewholebody",
    "sdpose",
    "openpifpaf",
    "openpose",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep missing-hand percentage across pose estimators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    path_group = parser.add_mutually_exclusive_group(required=True)
    path_group.add_argument(
        "--base-path",
        help=(
            "Base directory containing one sub-directory per estimator "
            "(i.e. <base-path>/<estimator>/)."
        ),
    )
    path_group.add_argument(
        "--path",
        help="Single .pose file or directory used for all estimators.",
    )

    parser.add_argument(
        "--estimators",
        nargs="+",
        default=DEFAULT_ESTIMATORS,
        metavar="ESTIMATOR",
        help=(
            "Pose estimator names to evaluate. "
            f"Defaults to: {' '.join(DEFAULT_ESTIMATORS)}"
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=1e-12,
        help=(
            "Keypoints with confidence below this value are considered missing. "
            "Default 1e-12 (effectively conf==0)."
        ),
    )
    parser.add_argument(
        "--missing-thresholds",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        metavar="PCT",
        help=(
            "Missing-ratio thresholds in percent to sweep over. "
            "Default: 10 20 30 40 50 60 70 80 90 100"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print frame-level debug info.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for estimator in args.estimators:
        if args.base_path:
            path = os.path.join(args.base_path, estimator)
        else:
            path = args.path

        if not os.path.exists(path):
            print(f"\n[SKIP] Path does not exist: {path}")
            continue

        try:
            pose = load_pose_or_directory(path)
        except Exception as exc:
            print(f"\n[ERROR] Could not load {path}: {exc}")
            continue

        fmt = detect_known_pose_format(pose)
        total_frames = pose.body.data.shape[0]

        print("\n" + "=" * 80)
        print(f"Pose Estimator : {estimator}")
        print(f"Path           : {path}")
        print(f"Detected format: {fmt}")
        print(f"Total frames   : {total_frames}")

        for th_pct in args.missing_thresholds:
            missing_ratio_threshold = th_pct / 100.0

            left = compute_signing_missing_hand_percentage(
                pose, "LEFT", args.confidence_threshold, missing_ratio_threshold,
                verbose=args.verbose,
            )
            right = compute_signing_missing_hand_percentage(
                pose, "RIGHT", args.confidence_threshold, missing_ratio_threshold,
                verbose=args.verbose,
            )
            both = compute_signing_missing_hand_percentage(
                pose, "BOTH", args.confidence_threshold, missing_ratio_threshold,
                verbose=args.verbose,
            )

            print(
                f"\n  missing_threshold = {th_pct:>3d}%  "
                f"(ratio={missing_ratio_threshold:.2f})"
            )
            print(f"    Left : {left:.6f}")
            print(f"    Right: {right:.6f}")
            print(f"    Both : {both:.6f}")


if __name__ == "__main__":
    main()

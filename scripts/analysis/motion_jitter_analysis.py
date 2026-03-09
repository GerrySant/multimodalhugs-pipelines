"""
Qualitative analysis: motion energy and jitter across pose estimators.

Computes per-sequence motion energy, acceleration jitter, and jerk jitter,
then optionally runs paired significance tests between a reference estimator
and the rest.

Expected directory structure
-----------------------------
The base directory must contain one sub-directory per pose estimator.
Each sub-directory may hold either a single .pose file or multiple .pose files
(one per video / sentence), which are processed independently (not concatenated):

    <base-path>/
    ├── mediapipe/
    │   ├── sentence_001.pose
    │   ├── sentence_002.pose
    │   └── ...
    ├── alphapose_133/
    │   ├── sentence_001.pose
    │   └── ...
    └── openpose/
        └── poses.pose

Usage examples
--------------

  # Basic run with default estimators and regions
  python motion_jitter_analysis.py --base-path /path/to/qualitative_evaluation/media/phoenix

  # Restrict estimators and regions
  python motion_jitter_analysis.py \\
      --base-path /path/to/... \\
      --estimators mediapipe alphapose_133 openpose \\
      --regions all hands

  # Add significance tests against a reference estimator
  python motion_jitter_analysis.py \\
      --base-path /path/to/... \\
      --reference alphapose_133 \\
      --sig-key jerk_mean_time
"""

import argparse
import glob
import os

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon

from pose_format import Pose
from pose_format.utils.generic import pose_hide_legs


# =========================================================
# Loading
# =========================================================

def load_pose(pose_file, normalize=False, reduce_legs=False):
    with open(pose_file, "rb") as f:
        pose = Pose.read(f)

    if reduce_legs:
        pose_hide_legs(pose)

    if normalize:
        pose = pose.normalize()

    return pose


def pose_to_tensor(pose):
    tensor = pose.torch().body.data.zero_filled()
    T, P, N, D = tensor.shape
    if P > 1:
        tensor = tensor[:, :1]
    return tensor.view(T, N, D)[..., :2]


# =========================================================
# Region filtering
# =========================================================

def filter_region(pose, region="all"):
    """
    Keep only the components relevant to `region`.

    Parameters
    ----------
    region : {"all", "hands", "face"}
    """
    if region == "all":
        return pose

    to_remove = []
    for component in pose.header.components:
        name = component.name.lower()
        if region == "hands" and "hand" not in name:
            to_remove.append(component.name)
        elif region == "face" and "face" not in name:
            to_remove.append(component.name)

    if to_remove:
        pose = pose.remove_components(components_to_remove=to_remove)

    return pose


# =========================================================
# Derivatives
# =========================================================

def velocity(x):
    return x[1:] - x[:-1]


def acceleration(x):
    v = velocity(x)
    return v[1:] - v[:-1]


def jerk(x):
    return x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]


# =========================================================
# Per-frame jitter
# =========================================================

def per_frame_accel_jitter(x):
    a = acceleration(x)
    return torch.linalg.norm(a, dim=-1).mean(dim=1)


def per_frame_jerk_jitter(x):
    j = jerk(x)
    return torch.linalg.norm(j, dim=-1).mean(dim=1)


# =========================================================
# Motion energy
# =========================================================

def motion_energy(x):
    v = velocity(x)
    return torch.linalg.norm(v, dim=-1).mean()


# =========================================================
# Sequence-level analysis
# =========================================================

def analyze_sequence(pose_path, region="all", normalize=True):
    pose = load_pose(pose_path, normalize=normalize, reduce_legs=True)
    pose = filter_region(pose, region=region)
    x = pose_to_tensor(pose)

    accel_pf = per_frame_accel_jitter(x)
    jerk_pf = per_frame_jerk_jitter(x)

    return {
        "motion_energy": motion_energy(x).item(),
        "accel_mean_time": accel_pf.mean().item(),
        "accel_std_time": accel_pf.std().item(),
        "jerk_mean_time": jerk_pf.mean().item(),
        "jerk_std_time": jerk_pf.std().item(),
    }


# =========================================================
# Multi-sequence analysis
# =========================================================

def analyze_path(path, region="all"):
    """
    Analyze all .pose files under `path` (or `path` itself if it is a file).
    Returns a list of per-sequence result dicts.
    """
    if os.path.isdir(path):
        pose_files = sorted(glob.glob(os.path.join(path, "*.pose")))
    else:
        pose_files = [path]

    return [analyze_sequence(pf, region=region) for pf in pose_files]


# =========================================================
# Significance testing
# =========================================================

def significance_test(results_a, results_b, key="jerk_mean_time"):
    values_a = np.array([r[key] for r in results_a])
    values_b = np.array([r[key] for r in results_b])

    _, p_t = ttest_rel(values_a, values_b)
    _, p_w = wilcoxon(values_a, values_b)

    return {
        "mean_A": values_a.mean(),
        "mean_B": values_b.mean(),
        "p_ttest": p_t,
        "p_wilcoxon": p_w,
    }


# =========================================================
# CLI
# =========================================================

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

DEFAULT_REGIONS = ["all", "hands", "face"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Motion energy and jitter analysis across pose estimators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-path",
        required=True,
        help=(
            "Base directory containing one sub-directory per estimator "
            "(i.e. <base-path>/<estimator>/)."
        ),
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=DEFAULT_ESTIMATORS,
        metavar="ESTIMATOR",
        help=f"Pose estimator names to evaluate. Defaults to: {' '.join(DEFAULT_ESTIMATORS)}",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=DEFAULT_REGIONS,
        choices=DEFAULT_REGIONS,
        metavar="REGION",
        help="Body regions to analyse. Choices: all hands face. Default: all three.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        metavar="ESTIMATOR",
        help=(
            "Reference estimator for paired significance tests. "
            "If omitted, significance tests are skipped."
        ),
    )
    parser.add_argument(
        "--sig-key",
        default="jerk_mean_time",
        choices=["motion_energy", "accel_mean_time", "accel_std_time", "jerk_mean_time", "jerk_std_time"],
        help="Metric used for significance testing. Default: jerk_mean_time.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable pose normalization before analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    normalize = not args.no_normalize

    all_results = {region: {} for region in args.regions}

    for region in args.regions:
        print("\n=================================================")
        print(f"REGION: {region.upper()}")
        print("=================================================")

        for estimator in args.estimators:
            path = os.path.join(args.base_path, estimator)

            if not os.path.exists(path):
                print(f"\n[SKIP] Path does not exist: {path}")
                continue

            try:
                results = analyze_path(path, region=region)
            except Exception as exc:
                print(f"\n[ERROR] {estimator}: {exc}")
                continue

            all_results[region][estimator] = results

            motion = np.mean([r["motion_energy"] for r in results]) * 100
            accel_mean = np.mean([r["accel_mean_time"] for r in results]) * 100
            accel_std = np.mean([r["accel_std_time"] for r in results]) * 100
            jerk_mean = np.mean([r["jerk_mean_time"] for r in results]) * 100
            jerk_std = np.mean([r["jerk_std_time"] for r in results]) * 100

            print(f"\n--- {estimator.upper()} ---")
            print(f"Motion Energy        : {motion:.5f}")
            print(f"Acceleration Jitter  : {accel_mean:.5f} ± {accel_std:.5f}")
            print(f"Jerk Jitter          : {jerk_mean:.5f} ± {jerk_std:.5f}")

    # --------------------------------------------------
    # Significance tests
    # --------------------------------------------------
    if args.reference is None:
        return

    for region in args.regions:
        region_results = all_results[region]

        if args.reference not in region_results:
            print(f"\n[SKIP] Reference estimator '{args.reference}' not found for region '{region}'.")
            continue

        print("\n=================================================")
        print(f"SIGNIFICANCE TEST — REGION: {region.upper()}")
        print(f"Key: {args.sig_key}")
        print("=================================================")

        for estimator in args.estimators:
            if estimator == args.reference or estimator not in region_results:
                continue

            stats = significance_test(
                region_results[args.reference],
                region_results[estimator],
                key=args.sig_key,
            )

            print(f"\n{args.reference} vs {estimator}")
            print(f"  Mean {args.reference:<20}: {stats['mean_A']:.5f}")
            print(f"  Mean {estimator:<20}: {stats['mean_B']:.5f}")
            print(f"  Paired t-test  p-value : {stats['p_ttest']:.6f}")
            print(f"  Wilcoxon       p-value : {stats['p_wilcoxon']:.6f}")


if __name__ == "__main__":
    main()

"""
Qualitative analysis: motion energy and jitter across pose estimators.

For each video sequence, computes the mean of per-time-step motion energy,
acceleration jitter, and jerk jitter.  This yields one scalar per sequence
per metric.  The distribution of those per-sequence values is then plotted
(one violin / box per estimator) and summary statistics (mean, median, std)
are printed.

Expected directory structure
-----------------------------
    <base-path>/
    ├── mediapipe/
    │   ├── sentence_001.pose
    │   └── ...
    ├── alphapose_136/
    │   └── ...
    └── openpose/
        └── poses.pose

Usage
-----
  python motion_jitter_plots.py --base-path /path/to/qualitative_evaluation/media/phoenix
  python motion_jitter_plots.py --base-path /path/to/... --plot-type violin
  python motion_jitter_plots.py --base-path /path/to/... --plot-type box
"""

import argparse
import glob
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
# Per-time-step jitter (averaged over keypoints)
# =========================================================

def per_timestep_accel_jitter(x):
    a = acceleration(x)
    return torch.linalg.norm(a, dim=-1).mean(dim=1)

def per_timestep_jerk_jitter(x):
    j = jerk(x)
    return torch.linalg.norm(j, dim=-1).mean(dim=1)


# =========================================================
# Motion energy
# =========================================================

def motion_energy(x):
    v = velocity(x)
    return torch.linalg.norm(v, dim=-1).mean()


# =========================================================
# Sequence-level analysis  (one scalar per sequence per metric)
# =========================================================

def analyze_sequence(pose_path, region="all", normalize=True):
    """
    Return a dict with one scalar per metric for this sequence.
    Each scalar is the temporal mean of the per-time-step values.
    """
    pose = load_pose(pose_path, normalize=normalize, reduce_legs=True)
    pose = filter_region(pose, region=region)
    x = pose_to_tensor(pose)

    accel_pf = per_timestep_accel_jitter(x)
    jerk_pf = per_timestep_jerk_jitter(x)

    return {
        "motion_energy": motion_energy(x).item(),
        "accel_mean_time": accel_pf.mean().item(),
        "accel_std_time": accel_pf.std().item(),
        "jerk_mean_time": jerk_pf.mean().item(),
        "jerk_std_time": jerk_pf.std().item(),
    }


def analyze_path(path, region="all", normalize=True):
    """
    Analyze all .pose files under `path`.
    Returns a list of per-sequence result dicts.
    """
    if os.path.isdir(path):
        pose_files = sorted(glob.glob(os.path.join(path, "*.pose")))
    else:
        pose_files = [path]
    return [analyze_sequence(pf, region=region, normalize=normalize)
            for pf in pose_files]


# =========================================================
# Plotting
# =========================================================

ESTIMATOR_COLORS = {
    "mediapipe":        "#E63946",
    "alphapose_133":    "#F4A261",
    "alphapose_136":    "#F4A261",
    "sapiens":          "#2A9D8F",
    "smplest_x":        "#264653",
    "mmposewholebody":  "#E9C46A",
    "sdpose":           "#7209B7",
    "openpifpaf":       "#3A86FF",
    "openpose":         "#06D6A0",
}

METRIC_LABELS = {
    "motion_energy":   "Motion Energy (×100)",
    "accel_mean_time": "Acceleration Jitter (×100)",
    "jerk_mean_time":  "Jerk Jitter (×100)",
}

METRICS = list(METRIC_LABELS.keys())


def plot_distributions(all_results, estimators, regions, plot_type="violin",
                       output_dir="plots"):
    """
    For every (metric, region) pair, plot the distribution of per-sequence
    values across estimators, with both mean and median markers.
    """
    os.makedirs(output_dir, exist_ok=True)

    for region in regions:
        region_data = all_results[region]
        present_estimators = [e for e in estimators if e in region_data]

        if not present_estimators:
            continue

        for metric in METRICS:
            fig, ax = plt.subplots(
                figsize=(max(10, len(present_estimators) * 1.4), 6))

            data_lists = []
            colors = []
            labels = []

            for est in present_estimators:
                values = np.array([r[metric] for r in region_data[est]]) * 100
                data_lists.append(values)
                colors.append(ESTIMATOR_COLORS.get(est, "#888888"))
                labels.append(est.replace("_", " ").title())

            positions = np.arange(1, len(present_estimators) + 1)

            if plot_type == "violin":
                parts = ax.violinplot(data_lists, positions=positions,
                                      showmeans=False, showmedians=False,
                                      showextrema=False)
                for i, body in enumerate(parts["bodies"]):
                    body.set_facecolor(colors[i])
                    body.set_edgecolor("black")
                    body.set_alpha(0.7)
                    body.set_linewidth(0.8)

                # Draw median and mean markers explicitly
                for i, vals in enumerate(data_lists):
                    med = np.median(vals)
                    mn = np.mean(vals)
                    # Median: white horizontal bar
                    ax.hlines(med, positions[i] - 0.15, positions[i] + 0.15,
                              color="white", linewidth=2.0, zorder=5)
                    # Mean: black diamond
                    ax.scatter(positions[i], mn, color="black",
                               marker="D", s=40, zorder=6)

                # Overlay individual points (jittered)
                for i, vals in enumerate(data_lists):
                    jitter = np.random.default_rng(42).uniform(
                        -0.12, 0.12, size=len(vals))
                    ax.scatter(positions[i] + jitter, vals,
                               color=colors[i], edgecolors="black",
                               linewidths=0.4, s=18, alpha=0.6, zorder=3)

            elif plot_type == "box":
                bp = ax.boxplot(data_lists, positions=positions,
                                patch_artist=True, widths=0.55,
                                showfliers=False)
                for i, (box, median_line) in enumerate(
                        zip(bp["boxes"], bp["medians"])):
                    box.set_facecolor(colors[i])
                    box.set_alpha(0.7)
                    box.set_edgecolor("black")
                    median_line.set_color("white")
                    median_line.set_linewidth(1.5)
                for element in ["whiskers", "caps"]:
                    for item in bp[element]:
                        item.set_color("black")
                        item.set_linewidth(0.8)

                # Mean markers
                for i, vals in enumerate(data_lists):
                    ax.scatter(positions[i], vals.mean(), color="black",
                               marker="D", s=40, zorder=6)

                # Overlay individual points
                for i, vals in enumerate(data_lists):
                    jitter = np.random.default_rng(42).uniform(
                        -0.15, 0.15, size=len(vals))
                    ax.scatter(positions[i] + jitter, vals,
                               color=colors[i], edgecolors="black",
                               linewidths=0.4, s=18, alpha=0.5, zorder=3)

            elif plot_type == "strip":
                for i, vals in enumerate(data_lists):
                    jitter = np.random.default_rng(42).uniform(
                        -0.25, 0.25, size=len(vals))
                    ax.scatter(positions[i] + jitter, vals,
                               color=colors[i], edgecolors="black",
                               linewidths=0.4, s=24, alpha=0.6, zorder=3)
                    # Mean: black diamond
                    ax.scatter(positions[i], vals.mean(), color="black",
                               marker="D", s=60, zorder=5)
                    # Median: white diamond
                    ax.scatter(positions[i], np.median(vals), color="white",
                               edgecolors="black", linewidths=1.0,
                               marker="D", s=60, zorder=5)

            # Legend for mean / median markers
            ax.scatter([], [], color="black", marker="D", s=40,
                       label="Mean")
            ax.plot([], [], color="white", linewidth=2.0,
                    label="Median", marker="_", markersize=10,
                    markeredgecolor="white", linestyle="None")
            ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
            ax.set_title(
                f"{METRIC_LABELS[metric]}  —  Region: {region.upper()}",
                fontsize=14, fontweight="bold", pad=12)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.grid(axis="y", which="minor", alpha=0.15, linewidth=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            fname = f"{plot_type}_{metric}_{region}.pdf"
            fig.savefig(os.path.join(output_dir, fname), dpi=200,
                        bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {os.path.join(output_dir, fname)}")


# =========================================================
# CLI
# =========================================================

DEFAULT_ESTIMATORS = [
    "mediapipe",
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
        description="Plot per-sequence jitter distributions across pose estimators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-path", required=True,
        help="Base directory containing one sub-directory per estimator.",
    )
    parser.add_argument(
        "--estimators", nargs="+", default=DEFAULT_ESTIMATORS,
        metavar="ESTIMATOR",
        help=f"Pose estimators to evaluate. Default: {' '.join(DEFAULT_ESTIMATORS)}",
    )
    parser.add_argument(
        "--regions", nargs="+", default=DEFAULT_REGIONS,
        choices=DEFAULT_REGIONS, metavar="REGION",
        help="Body regions to analyse. Default: all three.",
    )
    parser.add_argument(
        "--plot-type", default="violin",
        choices=["violin", "box", "strip"],
        help="Type of distribution plot. Default: violin.",
    )
    parser.add_argument(
        "--output-dir", default="plots",
        help="Directory where plot PDFs are saved. Default: ./plots",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Disable pose normalization before analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    normalize = not args.no_normalize

    all_results = {region: {} for region in args.regions}

    for region in args.regions:
        print(f"\nAnalysing region: {region.upper()}")
        for estimator in args.estimators:
            path = os.path.join(args.base_path, estimator)
            if not os.path.exists(path):
                print(f"  [SKIP] {path}")
                continue
            try:
                results = analyze_path(path, region=region, normalize=normalize)
                all_results[region][estimator] = results
                print(f"  {estimator}: {len(results)} sequences")
            except Exception as exc:
                print(f"  [ERROR] {estimator}: {exc}")

    # --------------------------------------------------
    # Print summary statistics (mean, median, std across sequences)
    # --------------------------------------------------
    for region in args.regions:
        region_data = all_results[region]
        present = [e for e in args.estimators if e in region_data]
        if not present:
            continue

        print("\n=================================================")
        print(f"REGION: {region.upper()}")
        print("=================================================")

        for est in present:
            results = region_data[est]

            motion   = np.array([r["motion_energy"]   for r in results]) * 100
            accel_m  = np.array([r["accel_mean_time"]  for r in results]) * 100
            accel_s  = np.array([r["accel_std_time"]   for r in results]) * 100
            jerk_m   = np.array([r["jerk_mean_time"]   for r in results]) * 100
            jerk_s   = np.array([r["jerk_std_time"]    for r in results]) * 100

            print(f"\n--- {est.upper()} ({len(results)} sequences) ---")
            print(f"Motion Energy        : mean={motion.mean():.5f}  median={np.median(motion):.5f}  std={motion.std():.5f}")
            print(f"Acceleration Jitter  : mean={accel_m.mean():.5f}  median={np.median(accel_m):.5f}  std={accel_m.std():.5f}")
            print(f"  (std across time)  : mean={accel_s.mean():.5f}  median={np.median(accel_s):.5f}  std={accel_s.std():.5f}")
            print(f"Jerk Jitter          : mean={jerk_m.mean():.5f}  median={np.median(jerk_m):.5f}  std={jerk_m.std():.5f}")
            print(f"  (std across time)  : mean={jerk_s.mean():.5f}  median={np.median(jerk_s):.5f}  std={jerk_s.std():.5f}")

    # --------------------------------------------------
    # Generate plots
    # --------------------------------------------------
    print("\nGenerating plots …")
    plot_distributions(
        all_results,
        estimators=args.estimators,
        regions=args.regions,
        plot_type=args.plot_type,
        output_dir=args.output_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()

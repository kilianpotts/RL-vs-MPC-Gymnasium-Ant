"""
vis_device_benchmark.py — Visualize CPU vs GPU training speed benchmark.

Usage:
    python3 vis_device_benchmark.py path/to/device_benchmark_results.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Visualize CPU vs GPU benchmark results")
    parser.add_argument("csv", type=Path, help="Path to device_benchmark_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    algos = df["algo"].unique()
    fig, axes = plt.subplots(1, len(algos), figsize=(5 * len(algos), 4), sharey=False)
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        data = df[df["algo"] == algo]

        means = data.groupby("device")["steps_per_second"].mean()
        stds  = data.groupby("device")["steps_per_second"].std()

        devices = means.index.tolist()
        x = range(len(devices))

        ax.bar(x, means.values, yerr=stds.values, capsize=6,
               color=["#4878CF", "#6ACC65"], alpha=0.8, width=0.4)

        # individual seed dots
        for i, device in enumerate(devices):
            vals = data[data["device"] == device]["steps_per_second"]
            ax.scatter([i] * len(vals), vals, color="black", zorder=3, s=30)

        # speedup annotation
        if "cpu" in means and "auto" in means:
            speedup = means["auto"] / means["cpu"]
            ax.text(0.98, 0.95, f"GPU speedup: {speedup:.2f}x",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        ax.set_xticks(list(x))
        ax.set_xticklabels([d.upper() for d in devices])
        ax.set_title(algo.upper())
        ax.set_ylabel("steps / second")
        ax.set_xlabel("device")

    fig.suptitle("CPU vs GPU training speed")
    plt.tight_layout()

    out_path = args.csv.parent / "device_benchmark.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
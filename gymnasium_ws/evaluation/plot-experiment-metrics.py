"""
plot_metrics.py — Plot rolling average reward over time for one or more experiments.

Usage:
    python plot_metrics.py results/sac-multi-policy-flat/metrics.csv
    python plot_metrics.py results/*/metrics.csv --window 50
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

CMD_COLORS = {
    "stand":   "#aec7e8",
    "forward": "#98df8a",
    "left":    "#ffbb78",
    "right":   "#f7b6d2",
}

LINE_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def shade_commands(ax, df):
    prev_cmd, start = df["command"].iloc[0], df["timestep"].iloc[0]
    for _, row in df.iterrows():
        if row["command"] != prev_cmd:
            ax.axvspan(start, row["timestep"], color=CMD_COLORS.get(prev_cmd, "#dddddd"), alpha=0.3)
            start, prev_cmd = row["timestep"], row["command"]
    ax.axvspan(start, df["timestep"].iloc[-1], color=CMD_COLORS.get(prev_cmd, "#dddddd"), alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(12, 5))

    first_df = pd.read_csv(args.csvs[0])
    shade_commands(ax, first_df)

    for i, path in enumerate(args.csvs):
        df = pd.read_csv(path)
        rolled = df["reward"].rolling(args.window).mean()
        ax.plot(df["timestep"], rolled, label=path.parent.name,
                color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=1.5)

    cmd_patches = [
        mpatches.Patch(color=c, alpha=0.4, label=cmd)
        for cmd, c in CMD_COLORS.items()
        if cmd in first_df["command"].values
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + cmd_patches, labels + [p.get_label() for p in cmd_patches],
              loc="upper right", fontsize=8)

    ax.set_title(f"Average reward over time (window={args.window})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward (rolling mean)")
    plt.tight_layout()
    plt.savefig("reward_plot.png", dpi=150)
    print("Saved reward_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
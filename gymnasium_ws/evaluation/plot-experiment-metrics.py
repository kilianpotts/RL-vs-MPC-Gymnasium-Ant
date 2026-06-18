"""Plot reward traces for one or more experiment `metrics.csv` files."""

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
    """Draw background spans that indicate active command segments."""
    prev_cmd, start = df["command"].iloc[0], df["timestep"].iloc[0]

    for _, row in df.iterrows():
        if row["command"] != prev_cmd:
            ax.axvspan(
                start,
                row["timestep"],
                color=CMD_COLORS.get(prev_cmd, "#dddddd"),
                alpha=0.65,
            )
            start, prev_cmd = row["timestep"], row["command"]

    ax.axvspan(
        start,
        df["timestep"].iloc[-1],
        color=CMD_COLORS.get(prev_cmd, "#dddddd"),
        alpha=0.65,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    # ---- styling (must be before plotting) ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(6, 4))

    # ---- data ----
    first_df = pd.read_csv(args.csvs[0])
    shade_commands(ax, first_df)

    curve_handles = []
    curve_labels = []

    for i, path in enumerate(args.csvs):
        df = pd.read_csv(path)
        rolled = df["reward"].rolling(args.window).mean()

        line, = ax.plot(
            df["timestep"],
            rolled,
            label=path.parent.name,
            color=LINE_COLORS[i % len(LINE_COLORS)],
            linewidth=1.5,
        )

        curve_handles.append(line)
        curve_labels.append(path.parent.name)

    # ---- command legend (left) ----
    cmd_patches = [
        mpatches.Patch(color=c, alpha=0.65, label=cmd)
        for cmd, c in CMD_COLORS.items()
        if cmd in first_df["command"].values
    ]

    leg1 = ax.legend(
        handles=cmd_patches,
        loc="lower left",
        ncol=2,  # <-- makes it 2x2
        fontsize=8,
        frameon=True,
        edgecolor="black",
        framealpha=0.9,
        title_fontsize=9,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    ax.add_artist(leg1)

    # ---- curve legend (right) ----
    ax.legend(
        handles=curve_handles,
        labels=curve_labels,
        loc="lower right",
        fontsize=8,
        frameon=True,
        edgecolor="black",
        framealpha=0.9,
        title_fontsize=9,
    )

    # ---- labels ----
    ax.set_title(f"Average reward over time (window={args.window})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward (rolling mean)")

    plt.tight_layout()
    plt.savefig(
        "/workspaces/gymnasium_ws/evaluation/plots/sac-single-vs-multi-rough.pdf",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved sac-single-vs-multi-rough.pdf")
    plt.show()


if __name__ == "__main__":
    main()
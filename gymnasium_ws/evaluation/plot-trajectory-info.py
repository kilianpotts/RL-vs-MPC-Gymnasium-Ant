import argparse
from pathlib import Path
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.transform as transform
import yaml

CMD_COLORS = {
    "stand":   "#aec7e8",
    "forward": "#98df8a",
    "left":    "#ffbb78",
    "right":   "#f7b6d2",
}

LINE_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_command_zones(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    zones = []
    current_idx = 0
    for block in config["command_sequence"]:
        cmd = block["command"]
        steps = block["steps"]
        zones.append({
            "command": cmd,
            "start": current_idx,
            "end": current_idx + steps
        })
        current_idx += steps

    return zones


def shade_background(axs, zones, max_len):
    for ax in axs:
        for zone in zones:
            if zone["start"] >= max_len:
                break
            end = min(zone["end"], max_len - 1)
            ax.axvspan(
                zone["start"],
                end,
                color=CMD_COLORS.get(zone["command"], "#dddddd"),
                alpha=0.65,
            )


def calculate_tum_metrics(df):
    z_height = df["tz"].values

    quats = df[["qx", "qy", "qz", "qw"]].values
    r = transform.Rotation.from_quat(quats)
    euler_angles_rad = r.as_euler("xyz", degrees=False)

    roll = np.degrees(np.unwrap(euler_angles_rad[:, 0]))
    pitch = np.degrees(np.unwrap(euler_angles_rad[:, 1]))

    return z_height, roll, pitch


def main():

    plt.rcParams.update({
    "font.family": "serif",

    # base text
    "font.size": 16,

    # axes
    "axes.labelsize": 18,
    "axes.titlesize": 18,

    # ticks
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,

    # legend
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
})
    parser = argparse.ArgumentParser()
    parser.add_argument("tum_files", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default="experiment_config.yaml")
    args = parser.parse_args()

    zones = load_command_zones(args.config)

    fig, axs = plt.subplots(
        3, 1,
        figsize=(10.5, 7.2),  # slightly taller to match bigger text
        sharex=True
    )
    metrics_labels = ["Z-Height (m)", "Roll (deg)", "Pitch (deg)"]

    max_rows = 0
    unique_commands_in_run = set()
    tum_cols = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]

    for file_idx, path in enumerate(args.tum_files):
        df = pd.read_csv(path, sep=r"\s+", comment="#", names=tum_cols)
        max_rows = max(max_rows, len(df))

        z, roll, pitch = calculate_tum_metrics(df)
        x_axis = np.arange(len(df))

        color = LINE_COLORS[file_idx % len(LINE_COLORS)]
        label = path.parent.name if path.parent.name else path.name

        axs[0].plot(x_axis, z, color=color, label=label, linewidth=1.5)
        axs[1].plot(x_axis, roll, color=color, linewidth=1.5)
        axs[2].plot(x_axis, pitch, color=color, linewidth=1.5)

    shade_background(axs, zones, max_rows)

    for zone in zones:
        if zone["start"] < max_rows:
            unique_commands_in_run.add(zone["command"])

    for idx, ax in enumerate(axs):
        ax.set_ylabel(metrics_labels[idx])
        ax.grid(True, alpha=0.3, linestyle="--")

    axs[-1].set_xlabel("Steps (Row Index)")

    # ---- build single legend (figure-level, top center) ----
    file_handles, file_labels = axs[0].get_legend_handles_labels()

    cmd_patches = [
        mpatches.Patch(
            color=CMD_COLORS[cmd],
            alpha=0.65,
            label=cmd
        )
        for cmd in CMD_COLORS
        if cmd in unique_commands_in_run
    ]

    fig.legend(
        handles=file_handles + cmd_patches,
        loc="upper center",
        ncol=3,
        fontsize=16,
        frameon=True,
        edgecolor="black",
        framealpha=0.9,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    # ---- leave space for legend at top ----
    plt.tight_layout(rect=[0, 0, 1, 0.89])

    plt.savefig(
        "/workspaces/gymnasium_ws/evaluation/plots/experiment-trajectory-rough.pdf",
        dpi=150,
        bbox_inches="tight"
    )

    print("Saved clean visualization to experiment-trajectory-rough.pdf")
    plt.show()


if __name__ == "__main__":
    main()
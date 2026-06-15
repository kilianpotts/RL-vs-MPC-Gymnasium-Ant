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
        zones.append({"command": cmd, "start": current_idx, "end": current_idx + steps})
        current_idx += steps
    return zones


def shade_background(axs, zones, max_len):
    for ax in axs:
        for zone in zones:
            if zone["start"] >= max_len:
                break
            end = min(zone["end"], max_len - 1)
            ax.axvspan(zone["start"], end, color=CMD_COLORS.get(zone["command"], "#dddddd"), alpha=0.3)


def calculate_tum_metrics(df):
    """Computes z-height, unwrapped roll, and unwrapped pitch from TUM data."""
    # 1. Z-Height
    z_height = df["tz"].values

    # 2. Orientation extraction (Roll and Pitch)
    quats = df[["qx", "qy", "qz", "qw"]].values
    r = transform.Rotation.from_quat(quats)
    euler_angles_rad = r.as_euler("xyz", degrees=False)
    
    # Extract roll (index 0) and unwrap
    roll_rad = np.unwrap(euler_angles_rad[:, 0])
    roll = np.degrees(roll_rad)

    # Extract pitch (index 1) and unwrap
    pitch_rad = np.unwrap(euler_angles_rad[:, 1])
    pitch = np.degrees(pitch_rad)

    return z_height, roll, pitch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tum_files", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default="experiment_config.yaml")
    args = parser.parse_args()

    zones = load_command_zones(args.config)

    # 3 stacked subplots sharing the X-axis
    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    metrics_labels = ["Z-Height (m)", "Roll (Unwrapped deg)", "Pitch (Unwrapped deg)"]

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

        axs[0].plot(x_axis, z, color=color, label=label, linewidth=1.2)
        axs[1].plot(x_axis, roll, color=color, linewidth=1.2)
        axs[2].plot(x_axis, pitch, color=color, linewidth=1.2)

    shade_background(axs, zones, max_rows)
    for zone in zones:
        if zone["start"] < max_rows:
            unique_commands_in_run.add(zone["command"])

    for idx, ax in enumerate(axs):
        ax.set_ylabel(metrics_labels[idx])
        ax.grid(True, alpha=0.3, linestyle="--")

    axs[-1].set_xlabel("Steps (Row Index)")

    # Legend construction
    file_handles, file_labels = axs[0].get_legend_handles_labels()
    cmd_patches = [
        mpatches.Patch(color=CMD_COLORS[cmd], alpha=0.4, label=cmd)
        for cmd in CMD_COLORS if cmd in unique_commands_in_run
    ]
    
    axs[0].legend(handles=file_handles + cmd_patches, labels=file_labels + [p.get_label() for p in cmd_patches],
                  loc="upper right", bbox_to_anchor=(1.15, 1.05), fontsize=8)

    plt.tight_layout()
    plt.savefig("tum_metrics_plot.png", dpi=150, bbox_inches="tight")
    print("Saved clean visualization to tum_metrics_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
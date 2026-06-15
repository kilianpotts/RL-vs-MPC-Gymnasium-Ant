"""
experiment.py — Scripted evaluation for single-policy vs multi-policy control.

Experiments:
    {sac,ppo}-{single,multi}-policy-{flat,rough}

Each experiment runs a fixed command sequence (defined in experiment_config.yaml),
logs per-step reward + timestep to metrics.csv, pose to trajectory.tum,
and saves a recording.mp4.

Usage:
    python experiment.py --config experiment_config.yaml --experiment sac-multi-policy-flat
    python experiment.py --config experiment_config.yaml --experiment ppo-single-policy-rough

Valid experiment names:
    sac-single-policy-flat   sac-single-policy-rough
    sac-multi-policy-flat    sac-multi-policy-rough
    ppo-single-policy-flat   ppo-single-policy-rough
    ppo-multi-policy-flat    ppo-multi-policy-rough
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml
from stable_baselines3 import PPO, SAC

from ant_rl.config import (
    ALGO_DEVICE,
    CMD_DIM,
    CMD_FORWARD,
    CMD_LEFT,
    CMD_RIGHT,
    CMD_STAND,
)
from ant_rl.env import CmdAnt
from ant_rl.rough_env import RoughCmdAnt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGORITHMS = {"sac": SAC, "ppo": PPO}

CMD_MAP = {
    "stand":   CMD_STAND,
    "forward": CMD_FORWARD,
    "left":    CMD_LEFT,
    "right":   CMD_RIGHT,
}

TERRAIN_SEED = 42
RECORD_FPS   = 30

# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_experiment(name: str) -> tuple[str, str, str]:
    """Parse 'sac-multi-policy-flat' -> ('sac', 'multi', 'flat')."""
    valid_algos    = {"sac", "ppo"}
    valid_policies = {"single", "multi"}
    valid_terrains = {"flat", "rough"}

    parts = name.split("-")
    # Expected format: algo-{single|multi}-policy-{flat|rough}
    if len(parts) != 4 or parts[2] != "policy":
        sys.exit(f"Invalid experiment name: '{name}'. "
                 f"Expected format: {{sac|ppo}}-{{single|multi}}-policy-{{flat|rough}}")

    algo, policy_type, _, terrain = parts

    if algo not in valid_algos:
        sys.exit(f"Unknown algo '{algo}'. Choose from: {valid_algos}")
    if policy_type not in valid_policies:
        sys.exit(f"Unknown policy type '{policy_type}'. Choose from: {valid_policies}")
    if terrain not in valid_terrains:
        sys.exit(f"Unknown terrain '{terrain}'. Choose from: {valid_terrains}")

    return algo, policy_type, terrain


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_single_model(config: dict, algo: str):
    path = Path(config["models"][algo]["single"])
    if not path.exists():
        sys.exit(f"Model not found: {path}")
    print(f"  Loading single-policy model: {path}")
    return ALGORITHMS[algo].load(str(path), device=ALGO_DEVICE[algo])


def load_multi_models(config: dict, algo: str) -> dict[str, tuple]:
    """Returns {command_name: (model, cmd_array)}."""
    paths = config["models"][algo]["multi"]
    model_map = {}
    loader = ALGORITHMS[algo]
    for tag, path_str in paths.items():
        path = Path(path_str)
        if not path.exists():
            sys.exit(f"Model not found for '{tag}': {path}")
        print(f"  Loading expert '{tag}': {path}")
        model_map[tag] = (
            loader.load(str(path), device=ALGO_DEVICE[algo]),
            CMD_MAP[tag].copy(),
        )
    return model_map


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(terrain: str, roughness: float, first_cmd: np.ndarray) -> CmdAnt:
    if terrain == "flat":
        return CmdAnt(
            command=first_cmd.copy(),
            render_mode="rgb_array",
        )
    else:
        return RoughCmdAnt(
            command=first_cmd.copy(),
            render_mode="rgb_array",
            terrain_roughness=roughness,
            terrain_seed=TERRAIN_SEED,
        )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def open_metrics_csv(path: Path):
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["timestep", "command", "reward"])
    return f, writer


def write_tum_pose(f, timestep: int, obs_raw: np.ndarray):
    """
    TUM format: timestamp tx ty tz qx qy qz qw
    Raw Ant-v5 obs layout (with positions):
      [0]=x [1]=y [2]=z [3]=qw [4]=qx [5]=qy [6]=qz ...
    """
    x, y, z       = obs_raw[0], obs_raw[1], obs_raw[2]
    qw, qx, qy, qz = obs_raw[3], obs_raw[4], obs_raw[5], obs_raw[6]
    f.write(f"{timestep} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    experiment_name: str,
    algo: str,
    policy_type: str,
    terrain: str,
    config: dict,
    out_dir: Path,
):
    roughness      = config["terrain"]["roughness"]
    cmd_sequence   = config["command_sequence"]

    # --- load models ---
    if policy_type == "single":
        single_model = load_single_model(config, algo)
        cmd_to_model = None  # same model for every command
    else:
        model_map    = load_multi_models(config, algo)
        cmd_to_model = {
            tuple(cmd.astype(int)): (model, cmd)
            for _, (model, cmd) in model_map.items()
        }
        single_model = None

    # --- build flat command+step schedule ---
    schedule: list[tuple[np.ndarray, str]] = []
    for entry in cmd_sequence:
        cmd_array = CMD_MAP[entry["command"]].copy()
        for _ in range(entry["steps"]):
            schedule.append((cmd_array, entry["command"]))

    total_steps = len(schedule)
    first_cmd   = schedule[0][0]

    # --- environment ---
    env = make_env(terrain, roughness, first_cmd)

    # --- output files ---
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path    = out_dir / "metrics.csv"
    trajectory_path = out_dir / "trajectory.tum"
    video_path      = out_dir / "recording.mp4"

    metrics_file, metrics_writer = open_metrics_csv(metrics_path)
    traj_file   = open(trajectory_path, "w")
    traj_file.write("# timestamp tx ty tz qx qy qz qw\n")

    frames = []

    # Raw obs is needed for TUM pose; we get it from env.env (the inner Ant-v5).
    # CmdAnt.reset/step returns the *wrapped* obs, so we access raw separately.
    def get_raw_obs() -> np.ndarray:
        return env.env.unwrapped.data.qpos  # positions: x,y,z,qw,qx,qy,qz,...

    print(f"\n=== {experiment_name} | {total_steps} steps ===")

    obs, _ = env.reset(seed=0)

    # Set initial model + command
    current_cmd      = schedule[0][0].copy()
    current_cmd_name = schedule[0][1]

    if policy_type == "multi":
        active_model, current_cmd = model_map.get(
            current_cmd_name,
            next(iter(model_map.values()))
        )
    else:
        active_model = single_model

    env.set_command(current_cmd)

    try:
        for step, (cmd_array, cmd_name) in enumerate(schedule):

            # Switch command (and expert if multi-policy)
            if not np.array_equal(cmd_array, current_cmd):
                current_cmd      = cmd_array.copy()
                current_cmd_name = cmd_name

                if policy_type == "multi":
                    key = tuple(current_cmd.astype(int))
                    if key in cmd_to_model:
                        active_model, current_cmd = cmd_to_model[key]

                env.set_command(current_cmd)

            obs[-CMD_DIM:] = current_cmd
            action, _      = active_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            # Log metrics
            metrics_writer.writerow([step, cmd_name, f"{reward:.6f}"])

            # Log TUM trajectory
            raw_qpos = get_raw_obs()
            write_tum_pose(traj_file, step, raw_qpos)

            # Record frame with overlaid command (top-left) and timestep (top-right)
            frame = env.render()
            if frame is not None:
                frame = frame.copy()
                h, w = frame.shape[:2]
                font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                label = cmd_name.upper()
                step_str = f"{step + 1}/{total_steps}"
                # top-left: command
                cv2.putText(frame, label, (10, 28), font, scale, (255, 255, 255), thickness)
                # top-right: timestep — measure text width to right-align
                (tw, _), _ = cv2.getTextSize(step_str, font, scale, thickness)
                cv2.putText(frame, step_str, (w - tw - 10, 28), font, scale, (255, 255, 255), thickness)
                frames.append(frame)

            if terminated or truncated:
                print(f"  [!] Episode ended at step {step}/{total_steps} "
                      f"({'terminated' if terminated else 'truncated'}). Stopping experiment.")
                break

            if (step + 1) % 100 == 0:
                print(f"  step {step + 1}/{total_steps}  cmd={cmd_name}  reward={reward:.3f}")

    finally:
        env.close()
        metrics_file.close()
        traj_file.close()

    # --- save video ---
    if frames:
        print(f"  Saving video ({len(frames)} frames) -> {video_path}")
        imageio.mimsave(str(video_path), frames, fps=RECORD_FPS)
    else:
        print("  Warning: no frames captured, video not saved.")

    print(f"  metrics    -> {metrics_path}")
    print(f"  trajectory -> {trajectory_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a scripted ant experiment (single/multi policy, flat/rough terrain)."
    )
    parser.add_argument(
        "--config",
        default="/workspaces/gymnasium_ws/experiment-config.yaml",
        help="Path to experiment_config.yaml",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name, e.g. sac-multi-policy-flat",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    algo, policy_type, terrain = parse_experiment(args.experiment)

    out_dir = Path(config["output_dir"]) / args.experiment

    print(f"Experiment : {args.experiment}")
    print(f"Algo       : {algo.upper()}")
    print(f"Policy     : {policy_type}")
    print(f"Terrain    : {terrain}")
    print(f"Output     : {out_dir}")

    run_experiment(
        experiment_name=args.experiment,
        algo=algo,
        policy_type=policy_type,
        terrain=terrain,
        config=config,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
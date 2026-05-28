"""
record_clips.py — Record MP4/GIF clips for expert Ant models.

Records four presentation clips:
  - stand
  - forward
  - left
  - right

Each clip loads the latest expert checkpoint for that command.

Usage:
  python3 record_clips.py --algo sac
  python3 record_clips.py --algo ppo

Recommended for PowerPoint:
  python3 record_clips.py --algo sac --duration 4 --fps 30 --no-gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO, SAC

from ant_rl.artifacts import latest_checkpoint
from ant_rl.config import (
    ALGO_DEVICE,
    CMD_DIM,
    CMD_STAND,
    CMD_FORWARD,
    CMD_LEFT,
    CMD_RIGHT,
)
from ant_rl.env import CmdAnt


ALGORITHMS = {
    "sac": SAC,
    "ppo": PPO,
}

COMMANDS = {
    "stand": CMD_STAND,
    "forward": CMD_FORWARD,
    "left": CMD_LEFT,
    "right": CMD_RIGHT,
}

OUTPUT_DIR = Path("artifacts/videos")


def load_expert_model(command_name: str, algo: str):
    """
    Load latest expert checkpoint for the given command.
    """
    checkpoint = latest_checkpoint(algo, command_name)

    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found for command '{command_name}' with algorithm '{algo}'.\n"
            f"Expected something like:\n"
            f"  artifacts/models/ant_cmd_{algo}_{command_name}_YYYYMMDD_HHMMSS.zip\n\n"
            f"Train it first with:\n"
            f"  python train_expert.py --command {command_name} --algo {algo}"
        )

    print(f"Loading {command_name} expert: {checkpoint}.zip")
    model = ALGORITHMS[algo].load(str(checkpoint), device=ALGO_DEVICE[algo])
    return model


def record_clip(
    command_name: str,
    command: np.ndarray,
    algo: str,
    duration: float,
    fps: int,
    seed: int,
    save_gif: bool,
):
    """
    Record one clip for one expert model and one command.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_expert_model(command_name, algo)

    env = CmdAnt(
        command=command.copy(),
        stage_probs={command_name: 1.0},
        render_mode="rgb_array",
    )

    frames = []
    total_frames = int(duration * fps)

    obs, _ = env.reset(seed=seed)
    env.set_command(command.copy())

    for _ in range(total_frames):
        # Ensure the observation always contains the correct command.
        obs[-CMD_DIM:] = command.copy()
        env.set_command(command.copy())

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()

        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            obs, _ = env.reset()
            env.set_command(command.copy())

    env.close()

    if not frames:
        raise RuntimeError(
            f"No frames were recorded for command '{command_name}'. "
            f"Check whether render_mode='rgb_array' works."
        )

    mp4_path = OUTPUT_DIR / f"{command_name}.mp4"
    gif_path = OUTPUT_DIR / f"{command_name}.gif"

    imageio.mimsave(mp4_path, frames, fps=fps)
    print(f"Saved MP4: {mp4_path}")

    if save_gif:
        gif_fps = min(fps, 15)
        step = max(1, round(fps / gif_fps))
        gif_frames = frames[::step]

        imageio.mimsave(gif_path, gif_frames, fps=gif_fps)
        print(f"Saved GIF: {gif_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Record MP4/GIF clips for Ant expert models."
    )

    parser.add_argument(
        "--algo",
        choices=["sac", "ppo"],
        default="sac",
        help="Algorithm checkpoints to load.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Clip duration in seconds.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment seed.",
    )

    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Only save MP4 files.",
    )

    args = parser.parse_args()

    print("\nRecording expert clips")
    print(f"  algorithm: {args.algo}")
    print(f"  duration:  {args.duration}s")
    print(f"  fps:       {args.fps}")
    print(f"  output:    {OUTPUT_DIR}")
    print()

    for command_name, command in COMMANDS.items():
        print(f"\nRecording command: {command_name}")
        record_clip(
            command_name=command_name,
            command=command,
            algo=args.algo,
            duration=args.duration,
            fps=args.fps,
            seed=args.seed,
            save_gif=not args.no_gif,
        )

    print(f"\nDone. Files saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
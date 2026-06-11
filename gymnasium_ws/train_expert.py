"""
train_expert.py — Train a single-command expert policy.

Usage:
    python3 train_expert.py --command stand
    python3 train_expert.py --command forward --algo ppo
    python3 train_expert.py --command stand --init-mode retrain
    python3 train_expert.py --command forward --init-mode previous
    python3 train_expert.py --command forward --init-mode resume
    python3 train_expert.py --command forward --algo ppo --terrain rough --terrain-roughness 0.35

Examples:
    # Start completely fresh, ignoring all checkpoints
    python3 train_expert.py --command stand --init-mode retrain

    # Continue training an existing expert model of the same command
    python3 train_expert.py --command forward --algo ppo --init-mode resume

    # Fine-tune a flat expert model on rough terrain
    python3 train_expert.py --command forward --terrain rough --terrain-roughness 0.35 --init-mode previous

    # Train turning experts
    python3 train_expert.py --command left --init-mode retrain
    python3 train_expert.py --command right --init-mode retrain

Initialization modes:
    retrain   Start a new model from scratch. No checkpoint is loaded.
    previous  Load the previous/fallback checkpoint.
              For rough terrain, this is the same command trained on flat terrain.
              For flat expert training, this usually only exists if a fallback was configured.
    resume    Continue from an existing checkpoint of the same command and terrain.
"""

import argparse
from ant_rl import train_expert
from ant_rl.config import TRAINING


def main():
    # Load defaults from config.yaml
    training_cfg = TRAINING

    parser = argparse.ArgumentParser(
        description="Train an expert policy for a single command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--command",
        type=str,
        default="stand",
        choices=["stand", "forward", "left", "right"],
        help="Command to train expert for (default: stand)",
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="Algorithm to use (default: sac)",
    )

    parser.add_argument(
        "--init-mode",
        type=str,
        default="resume",
        choices=["retrain", "previous", "resume"],
        help=(
            "How to initialize training: "
            "'retrain' starts fresh, "
            "'previous' loads the previous/fallback model, "
            "'resume' loads the current command checkpoint if available "
            "(default: resume)"
        ),
    )

    parser.add_argument(
        "--terrain",
        type=str,
        default="flat",
        choices=["flat", "rough"],
        help="Terrain type to use: flat or rough (default: flat)",
    )

    parser.add_argument(
        "--terrain-roughness",
        type=float,
        default=0.35,
        help="Rough terrain strength. Only used with --terrain rough (default: 0.35)",
    )

    args = parser.parse_args()

    train_expert(
        command=args.command,
        algo=args.algo,
        init_mode=args.init_mode,
        timesteps=training_cfg["timesteps"],
        n_envs=training_cfg["n_envs"],
        seed=training_cfg["seed"],
        terrain=args.terrain,
        terrain_roughness=args.terrain_roughness,
    )


if __name__ == "__main__":
    main()

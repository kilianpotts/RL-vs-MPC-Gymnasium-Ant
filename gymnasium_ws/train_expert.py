"""
train_expert.py — Train a single-command expert policy.

Usage:
    python3 train_expert.py --command stand
    python3 train_expert.py --command forward --algo ppo
    python3 train_expert.py --command stand --init-mode retrain
    python3 train_expert.py --command forward --init-mode resume

Examples:
    # Start completely fresh, ignoring all checkpoints
    python3 train_expert.py --command stand --init-mode retrain

    # Continue training an existing expert model of the same command
    python3 train_expert.py --command forward --algo ppo --init-mode resume

    # Train turning experts
    python3 train_expert.py --command left --init-mode retrain
    python3 train_expert.py --command right --init-mode retrain

Initialization modes:
    retrain   Start a new model from scratch. No checkpoint is loaded.
    previous  Load the previous/fallback checkpoint if one is configured/found.
    resume    Continue from an existing checkpoint of the same command.
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
            "'previous' loads the previous/fallback model if available, "
            "'resume' loads the current command checkpoint if available "
            "(default: resume)"
        ),
    )

    args = parser.parse_args()

    train_expert(
        command=args.command,
        algo=args.algo,
        init_mode=args.init_mode,
        timesteps=training_cfg["timesteps"],
        n_envs=training_cfg["n_envs"],
        seed=training_cfg["seed"],
    )


if __name__ == "__main__":
    main()
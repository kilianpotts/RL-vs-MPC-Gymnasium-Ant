"""
train_curriculum.py — Train a curriculum stage with multiple commands.

Usage:
    python3 train_curriculum.py --stage cur_stand
    python3 train_curriculum.py --stage cur_forward --algo ppo
    python3 train_curriculum.py --stage cur_stand --init-mode retrain
    python3 train_curriculum.py --stage cur_forward --init-mode previous
    python3 train_curriculum.py --stage cur_forward --init-mode resume

Examples:
    # Start completely fresh, ignoring all checkpoints
    python3 train_curriculum.py --stage cur_stand --algo ppo --init-mode retrain

    # Initialize from the previous/fallback stage
    python3 train_curriculum.py --stage cur_forward --algo ppo --init-mode previous

    # Continue training an existing model of the same stage
    python3 train_curriculum.py --stage cur_forward --algo ppo --init-mode resume

The stage name must match a name in config.yaml curriculum.stages, and
command probabilities are read from there.

Initialization modes:
    retrain   Start a new model from scratch. No checkpoint is loaded.
    previous  Load the previous/fallback checkpoint if one is configured/found.
    resume    Continue from an existing checkpoint of the same stage.
"""

import argparse

from ant_rl import train_curriculum
from ant_rl.config import TRAINING


def main():
    # Load defaults from config.yaml
    training_cfg = TRAINING

    parser = argparse.ArgumentParser(
        description="Train a curriculum stage with multiple commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="cur_stand",
        help="Curriculum stage name from config.yaml (default: cur_stand)",
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="Algorithm to use (default: sac)",
    )

    parser.add_argument(
    "--init-checkpoint",
    type=str,
    default=None,
    help="Path to a checkpoint to initialize from",
)

    parser.add_argument(
        "--init-mode",
        type=str,
        default="resume",
        choices=["retrain", "previous", "resume"],
        help=(
            "How to initialize training: "
            "'retrain' starts fresh, "
            "'previous' loads the previous/fallback stage if available, "
            "'resume' loads the current stage checkpoint if available "
            "(default: resume)"
        ),
    )

    args = parser.parse_args()

    train_curriculum(
        stage=args.stage,
        algo=args.algo,
        init_mode=args.init_mode,
        init_checkpoint=args.init_checkpoint,
        timesteps=training_cfg["timesteps"],
        n_envs=training_cfg["n_envs"],
        seed=training_cfg["seed"],
    )


if __name__ == "__main__":
    main()
"""
train_curriculum.py — Train a curriculum stage with multiple commands.

Usage:
    python train_curriculum.py --stage stand              # train stand stage (SAC)
    python train_curriculum.py --stage forward --algo ppo # train forward stage (PPO)
    python train_curriculum.py --stage stand --retrain    # train from scratch
    
Examples:
    python train_curriculum.py --stage stand
    python train_curriculum.py --stage forward --algo ppo --retrain

The stage name must match a name in config.yaml curriculum.stages, and
command probabilities are read from there.
"""

import sys
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
        default="stand",
        help="Curriculum stage name from config.yaml (default: stand)"
    )
    
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="Algorithm to use (default: sac)"
    )
    
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoints"
    )
    
    args = parser.parse_args()
    
    train_curriculum(
        stage=args.stage,
        algo=args.algo,
        retrain=args.retrain,
        timesteps=training_cfg["timesteps"],
        n_envs=training_cfg["n_envs"],
        seed=training_cfg["seed"],
    )


if __name__ == "__main__":
    main()

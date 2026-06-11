"""
train_curriculum.py — Train a curriculum stage with multiple commands.

Usage:
    python3 train_curriculum.py --stage cur_stand
    python3 train_curriculum.py --stage cur_forward --algo ppo
    python3 train_curriculum.py --stage cur_stand --retrain
    python3 train_curriculum.py --stage cur_stand --terrain rough --terrain-roughness 0.35

Examples:
    python3 train_curriculum.py --stage cur_stand --algo ppo --retrain
    python3 train_curriculum.py --stage cur_forward --algo ppo --terrain rough --terrain-roughness 0.35

The stage name must match a name in config.yaml curriculum.stages, and
command probabilities are read from there.
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
        default="stand",
        help="Curriculum stage name from config.yaml (default: stand)",
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="Algorithm to use (default: sac)",
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoints",
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

    train_curriculum(
        stage=args.stage,
        algo=args.algo,
        retrain=args.retrain,
        timesteps=training_cfg["timesteps"],
        n_envs=training_cfg["n_envs"],
        seed=training_cfg["seed"],
        terrain=args.terrain,
        terrain_roughness=args.terrain_roughness,
    )


if __name__ == "__main__":
    main()
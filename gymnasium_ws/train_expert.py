"""
train_expert.py — Train a single-command expert policy.

Usage:
    python3 train_expert.py --command stand
    python3 train_expert.py --command forward --algo ppo
    python3 train_expert.py --command stand --retrain
    python3 train_expert.py --command forward --algo ppo --terrain rough --terrain-roughness 0.35

Examples:
    python3 train_expert.py --command stand
    python3 train_expert.py --command forward --algo ppo --retrain
    python3 train_expert.py --command left
    python3 train_expert.py --command right
    python3 train_expert.py --command forward --terrain rough --terrain-roughness 0.35
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

    train_expert(
        command=args.command,
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

"""
train_expert.py — Train a single-command expert policy.

Usage:
    python train_expert.py --command stand              # train stand expert (SAC)
    python train_expert.py --command forward --algo ppo # train forward expert (PPO)
    python train_expert.py --command stand --retrain    # train from scratch
    
Examples:
    python train_expert.py --command stand
    python train_expert.py --command forward --algo ppo --retrain
    python train_expert.py --command left
    python train_expert.py --command right
"""

import sys
import argparse
from ant_rl import train_expert
from ant_rl.config import load_config, TRAINING

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
        help="Command to train expert for (default: stand)"
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
    
    train_expert(
        command=args.command,
        algo=args.algo,
        retrain=args.retrain,
        timesteps=training_cfg["timesteps"],
        n_envs=training_cfg["n_envs"],
        seed=training_cfg["seed"],
    )


if __name__ == "__main__":
    main()

"""
grid_search.py — Hyperparameter grid search for the stand expert policy.

Searches a small range around the default params to validate the config.yaml
defaults. Results are written to grid_search_results.csv.

Usage:
    python grid_search.py --algo sac
    python grid_search.py --algo ppo
    python grid_search.py --algo sac --seeds 3

# Hyperparameter search scope
#
# SAC:  learning_rate and batch_size are the primary drivers of sample efficiency
#       and gradient quality. All other params are either theoretically grounded
#       (gamma, tau), self-tuning (ent_coef="auto"), or infrastructure (buffer_size, device).
#
# PPO:  learning_rate, n_steps, and ent_coef are searched. learning_rate controls
#       update stability, n_steps determines the on-policy data horizon and interacts
#       with variance/bias in advantage estimation, and ent_coef=0.0 may underexplore
#       on Ant. clip_range (0.2), gae_lambda (0.95), and n_epochs (10) are excluded
#       as they are near-universally stable at their defaults across locomotion tasks.
"""

import csv
import argparse
import itertools
import numpy as np
from pathlib import Path
import ant_rl.config as cfg
from ant_rl import train_expert
from ant_rl.config import TRAINING

GRIDS = {
    "sac": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "batch_size":    [128, 256, 512],
    },
    "ppo": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "n_steps":       [1024, 2048, 4096],
        "ent_coef":      [0.0, 0.001, 0.01],
    },
}

OUTPUT_FILE = Path("grid_search_results.csv")


def run_grid(algo: str, seeds: list[int]):
    """Train/evaluate all grid points and write aggregate results to CSV."""
    grid = GRIDS[algo]
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))

    print(f"Running {len(combos)} configs × {len(seeds)} seeds = {len(combos) * len(seeds)} runs\n")

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", *keys, "seed", "final_mean_reward", "steps"])
        writer.writeheader()

        for combo in combos:
            params = dict(zip(keys, combo))

            for seed in seeds:
                print(f"  {params}  seed={seed}")

                model = train_expert(
                    command="stand",
                    algo=algo,
                    retrain=True,
                    seed=seed,
                    algo_kwargs_override=params,
                    timesteps=100000,
                )

                ep_rewards = evaluate(model, seed=seed)
                final_mean_reward = float(np.mean(ep_rewards))
                steps = model.num_timesteps

                row = {"algo": algo, **params, "seed": seed,
                       "final_mean_reward": final_mean_reward, "steps": steps}
                writer.writerow(row)
                f.flush()
                print(f"    → mean_reward={final_mean_reward:.2f}  steps={steps}")

    print(f"\nDone. Results written to {OUTPUT_FILE}")


def evaluate(model, n_episodes: int = 10, seed: int = 42) -> list[float]:
    """Run n_episodes with the trained model and return episode rewards."""
    from ant_rl.train import make_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    env = SubprocVecEnv([make_env({"stand": 1.0}, seed=seed)])
    rewards = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0]
        rewards.append(ep_reward)

    env.close()
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Grid search over SAC/PPO hyperparameters")
    parser.add_argument("--algo",  type=str, default="sac", choices=["sac", "ppo"])
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (default: 3)")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    run_grid(args.algo, seeds)


if __name__ == "__main__":
    main()
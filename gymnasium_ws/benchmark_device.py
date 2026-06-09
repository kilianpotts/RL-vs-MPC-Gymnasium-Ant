"""
benchmark_device.py — Compare CPU vs GPU training speed for SAC and PPO.

Usage:
    python3 benchmark_device.py
    python3 benchmark_device.py --seeds 2
"""

import csv
import time
import argparse
from pathlib import Path

import ant_rl.config as cfg
from ant_rl import train_expert
from ant_rl.config import TRAINING

OUTPUT_FILE = Path("device_benchmark_results.csv")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU training speed")
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    configs = [
        {"algo": "sac", "device": "cpu"},
        {"algo": "sac", "device": "auto"},
        {"algo": "ppo", "device": "cpu"},
        {"algo": "ppo", "device": "auto"},
    ]

    print(f"Running {len(configs)} configs × {args.seeds} seeds = {len(configs) * args.seeds} runs\n")

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "device", "seed", "steps", "wall_seconds", "steps_per_second"])
        writer.writeheader()

        for cfg_entry in configs:
            algo   = cfg_entry["algo"]
            device = cfg_entry["device"]

            for seed in seeds:
                print(f"  algo={algo}  device={device}  seed={seed}")

                original_device = cfg.ALGO_DEVICE[algo]
                cfg.ALGO_DEVICE[algo] = device

                try:
                    t0 = time.perf_counter()
                    model = train_expert(
                        command="stand",
                        algo=algo,
                        retrain=True,
                        seed=seed,
                        timesteps=TRAINING["timesteps"],
                    )
                    wall_seconds = time.perf_counter() - t0
                finally:
                    cfg.ALGO_DEVICE[algo] = original_device

                steps = model.num_timesteps
                sps   = steps / wall_seconds

                print(f"    → steps={steps}  time={wall_seconds:.1f}s  steps/s={sps:.0f}\n")

                writer.writerow({
                    "algo":             algo,
                    "device":           device,
                    "seed":             seed,
                    "steps":            steps,
                    "wall_seconds":     round(wall_seconds, 2),
                    "steps_per_second": round(sps, 1),
                })
                f.flush()

    print(f"Done. Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
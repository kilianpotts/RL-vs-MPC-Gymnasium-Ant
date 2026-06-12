"""
rough_inference.py — Single-model rough-terrain inference endpoint.

Run a trained policy in the rough Ant environment.

Usage:
    python3 rough_inference.py --model path/to/model.zip
    python3 rough_inference.py --model artifacts/models/ant_cmd_sac_forward_20260610_092049.zip
    python3 rough_inference.py --model artifacts/models/ant_cmd_sac_forward_20260610_092049.zip --terrain-roughness 0.5
    python3 rough_inference.py --model artifacts/models/ant_cmd_sac_forward_20260610_092049.zip --terrain-seed 123

Examples:
    # Run with default roughness from config.yaml
    python3 rough_inference.py --model artifacts/models/ant_cmd_sac_forward_20260610_092049.zip

    # Override roughness manually
    python3 rough_inference.py --model artifacts/models/ant_cmd_sac_forward_20260610_092049.zip --terrain-roughness 0.5

Arguments:
    --model              Path to the trained model .zip file.
    --terrain-roughness  Optional rough terrain strength. If omitted, config.yaml is used.
    --terrain-seed       Optional fixed seed for reproducible terrain.

Notes:
    This script does not train on rough terrain.
    It only tests already trained models on rough terrain.
"""

import argparse
import sys
from pathlib import Path

from ant_rl.rough_inference import run_inference, infer_algo_from_path


def main():
    parser = argparse.ArgumentParser(
        description="Run single-policy inference on rough terrain."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (.zip)",
    )

    parser.add_argument(
        "--terrain-roughness",
        type=float,
        default=None,
        help="Rough terrain strength. If omitted, config.yaml is used.",
    )

    parser.add_argument(
        "--terrain-seed",
        type=int,
        default=None,
        help="Optional fixed seed for reproducible rough terrain.",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}")

    algo = infer_algo_from_path(model_path)

    run_inference(
        path=model_path,
        algo=algo,
        terrain_roughness=args.terrain_roughness,
        terrain_seed=args.terrain_seed,
    )


if __name__ == "__main__":
    main()
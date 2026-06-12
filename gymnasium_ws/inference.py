"""
inference.py — Single-model inference endpoint.

Run a trained policy in the Ant environment.

Usage:
    python3 inference.py --model path/to/model.zip
    python3 inference.py --model artifacts/models/sac/cur_forward_flat/latest.zip
    python3 inference.py --model artifacts/models/sac/cur_forward_rough/latest.zip --terrain rough --terrain-roughness 0.35

Examples:
    # Run a flat-terrain model
    python3 inference.py --model artifacts/models/sac/cur_forward_flat/latest.zip

 
Arguments:
    --model              Path to the trained model .zip file.

"""

import argparse
import sys
from pathlib import Path

from ant_rl.inference import run_inference, infer_algo_from_path


def main():
    parser = argparse.ArgumentParser(
        description="Run single-policy inference with specified model."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (.zip)",
    )

    parser.add_argument(
        "--terrain",
        type=str,
        default="flat",
        choices=["flat", "rough"],
        help="Terrain type: flat or rough",
    )

    parser.add_argument(
        "--terrain-roughness",
        type=float,
        default=0.35,
        help="Rough terrain strength",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}")

    algo = infer_algo_from_path(model_path)

    run_inference(
        path=model_path,
        algo=algo,
    )


if __name__ == "__main__":
    main()
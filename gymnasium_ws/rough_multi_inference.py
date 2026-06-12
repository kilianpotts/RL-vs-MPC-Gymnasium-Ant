"""
rough_multi_inference.py — Multi-model rough-terrain inference endpoint.

Edit MODEL_PATHS below with your trained model .zip files.
The script delegates to ant_rl.rough_inference for rendering and expert switching.

Run:
    # Run with default roughness from config.yaml
    python3 rough_multi_inference.py

    # Override roughness manually
    python3 rough_multi_inference.py --terrain-roughness 0.5

    # Use fixed terrain seed
    python3 rough_multi_inference.py --terrain-seed 123

Notes:
    This script does not require rough-trained models.
    It is meant to test already trained models on rough terrain.

    You can use the same flat-trained models here to test robustness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ant_rl.rough_inference import (
    build_model_map_from_paths,
    run_inference_multi_paths,
    infer_algo_from_path,
)


# Placeholder filepaths: replace with your own model checkpoints.
MODEL_PATHS = {
    "stand": Path("/workspaces/gymnasium_ws/artifacts/models/ant_cmd_sac_stand_20260610_232545.zip"),
    "forward": Path("/workspaces/gymnasium_ws/artifacts/models/ant_cmd_sac_forward_20260610_092049.zip"),
    "left": Path("/workspaces/gymnasium_ws/artifacts/models/ant_cmd_sac_left_20260605_052733.zip"),
    "right": Path("/workspaces/gymnasium_ws/artifacts/models/ant_cmd_sac_right_20260603_143452.zip"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-model expert switching inference on rough terrain."
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

    # Check all paths exist
    for path in MODEL_PATHS.values():
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

    # Check all models use same algo
    algos = {infer_algo_from_path(path) for path in MODEL_PATHS.values()}
    if len(algos) > 1:
        raise ValueError(f"All models must use the same algorithm, found: {algos}")

    algo = algos.pop()

    model_map = build_model_map_from_paths(MODEL_PATHS, algo=algo)

    run_inference_multi_paths(
        model_map,
        terrain_roughness=args.terrain_roughness,
        terrain_seed=args.terrain_seed,
    )


if __name__ == "__main__":
    main()
"""
multi_inference.py — Multi-model inference endpoint.

Edit MODEL_PATHS below with your trained model .zip files.
The script delegates to ant_rl.inference for rendering and expert switching.

Run:
    # Run with flat terrain
    python3 multi_inference.py

"""

from __future__ import annotations

import argparse
from pathlib import Path

from ant_rl.inference import (
    build_model_map_from_paths,
    run_inference_multi_paths,
    infer_algo_from_path,
)


# Placeholder filepaths: replace with your own model checkpoints.
MODEL_PATHS = {
    "stand": Path("/workspaces/gymnasium_ws/artifacts/models/old_models/ant_cmd_sac_stand_20260610_232545.zip"),
    # "forward": Path("/workspaces/gymnasium_ws/artifacts/models/ant_cmd_sac_forward_20260528_093547.zip"),
    "forward": Path("/workspaces/gymnasium_ws/artifacts/models/ant_cmd_ppo_forward_20260614_063248.zip"),
    "left": Path("/workspaces/gymnasium_ws/artifacts/models/old_models/ant_cmd_sac_left_20260605_052733.zip"),
    "right": Path("/workspaces/gymnasium_ws/artifacts/models/old_models/ant_cmd_sac_right_20260603_143452.zip"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-model expert switching inference."
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
    )


if __name__ == "__main__":
    main()

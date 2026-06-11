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
        terrain=args.terrain,
        terrain_roughness=args.terrain_roughness,
    )


if __name__ == "__main__":
    main()
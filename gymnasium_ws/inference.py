"""
inference.py - single-model inference endpoint.

Usage:
  python3 inference.py --alg sac
  python3 inference.py --alg ppo

Behavior:
  - Chooses the latest checkpoint for the selected algorithm from artifacts/models.
  - Uses ant_rl.inference.run_inference(...) to run live keyboard control inference.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ant_rl.inference import run_inference

MODEL_DIR = Path("/workspaces/gymnasium_ws/artifacts/models")
MODEL_STEM = "ant_cmd"


def _find_latest_model(algo: str) -> tuple[Path, str]:
    pattern = re.compile(rf"^{MODEL_STEM}_{algo}_(?P<tag>.+)_(?P<date>\d{{8}}_\d{{6}})\.zip$")
    candidates: list[tuple[str, Path, str]] = []

    for path in MODEL_DIR.glob(f"{MODEL_STEM}_{algo}_*.zip"):
        match = pattern.match(path.name)
        if match:
            candidates.append((match.group("date"), path, match.group("tag")))

    if not candidates:
        raise FileNotFoundError(f"No {algo} model found under {MODEL_DIR}")

    candidates.sort(key=lambda x: x[0])
    _, latest_path, tag = candidates[-1]
    return latest_path, tag


def main():
	parser = argparse.ArgumentParser(description="Run single-model inference using the latest model checkpoint")
	parser.add_argument("--alg", choices=["sac", "ppo"], default="sac", help="Algorithm to load")
	args = parser.parse_args()

	algo = args.alg
	model_path, tag = _find_latest_model(algo)
	print(f"Loading latest {algo} model: {model_path}")
	print(f"Detected tag: {tag}")
	run_inference(model_tag=tag, algo=algo)


if __name__ == "__main__":
	main()

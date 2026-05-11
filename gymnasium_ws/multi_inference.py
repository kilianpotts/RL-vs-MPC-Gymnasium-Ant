"""
multi_inference.py - multi-model inference endpoint (no arguments).

Edit MODEL_PATHS below with your trained model .zip files.
The script delegates to ant_rl.inference module for rendering and expert switching.

Run:
  python3 multi_inference.py
"""

from __future__ import annotations

from pathlib import Path

from ant_rl.inference import build_model_map_from_paths, run_inference_multi_paths


# Placeholder filepaths: replace with your own model checkpoints.
MODEL_PATHS = {
	"stand": Path("/path/to/stand_model.zip"),
	"forward": Path("/path/to/forward_model.zip"),
	"left": Path("/path/to/left_model.zip"),
	"right": Path("/path/to/right_model.zip"),
}


def main():
	model_map = build_model_map_from_paths(MODEL_PATHS, algo="sac")
	run_inference_multi_paths(model_map)


if __name__ == "__main__":
	main()

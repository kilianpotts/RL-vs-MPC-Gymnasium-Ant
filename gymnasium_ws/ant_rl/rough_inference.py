"""Single-model and multi-model rough-terrain inference helpers."""

from __future__ import annotations

import time
import os
import sys
from pathlib import Path

from stable_baselines3 import PPO, SAC

from .config import (
    ALGO_DEVICE,
    CMD_DIM,
    CMD_FORWARD,
    CMD_LEFT,
    CMD_NAMES,
    CMD_RIGHT,
    CMD_STAND,
    DEFAULT_TERRAIN_ROUGHNESS,
)
from .rough_env import RoughCmdAnt

ALGORITHMS = {"sac": SAC, "ppo": PPO}

KEY_W = ord("w")
KEY_A = ord("a")
KEY_D = ord("d")
KEY_ESC = 27


def _get_cv2():
    try:
        os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
        os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

        import cv2

        return cv2
    except ImportError:
        sys.exit("opencv-python is required: pip install opencv-python")


def _render_frame(
    cv2,
    env,
    current_cmd,
    reward,
    terrain_roughness: float,
    suffix: str | None = None,
):
    frame = env.render()
    if frame is None:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cmd_name = CMD_NAMES.get(tuple(current_cmd.astype(int)), "?")

    cv2.putText(
        frame,
        f"CMD: {cmd_name}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"reward: {reward:.3f}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )

    cv2.putText(
        frame,
        f"roughness: {terrain_roughness:.2f}",
        (10, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 200, 255),
        1,
    )

    if suffix:
        cv2.putText(
            frame,
            suffix,
            (10, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 200, 255),
            1,
        )

    return frame


def _read_command(key):
    if key == KEY_W:
        return CMD_FORWARD.copy()
    if key == KEY_A:
        return CMD_LEFT.copy()
    if key == KEY_D:
        return CMD_RIGHT.copy()
    return None


def _run_loop(
    active_model,
    current_cmd,
    cmd_to_model,
    window_name: str,
    terrain_roughness: float | None = None,
    terrain_seed: int | None = None,
):
    cv2 = _get_cv2()

    if terrain_roughness is None:
        terrain_roughness = DEFAULT_TERRAIN_ROUGHNESS

    env = RoughCmdAnt(
        command=current_cmd,
        stage_probs={"stand": 1.0},
        render_mode="rgb_array",
        terrain_roughness=terrain_roughness,
        terrain_seed=terrain_seed,
    )

    key_timeout_s = 0.5
    last_key_time = 0.0

    try:
        obs, _ = env.reset(seed=0)
        env.set_command(current_cmd)

        while True:
            obs[-CMD_DIM:] = current_cmd

            action, _ = active_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            frame = _render_frame(
                cv2,
                env,
                current_cmd,
                reward,
                terrain_roughness=terrain_roughness,
                suffix="[multi-expert]" if cmd_to_model is not None else None,
            )

            if frame is not None:
                cv2.imshow(window_name, frame)

            key = cv2.waitKey(16) & 0xFF

            if key == KEY_ESC:
                break

            new_cmd = _read_command(key)

            if new_cmd is not None:
                current_cmd = new_cmd
                last_key_time = time.monotonic()

                if cmd_to_model is not None:
                    new_key = tuple(current_cmd.astype(int))
                    if new_key in cmd_to_model:
                        active_model, current_cmd = cmd_to_model[new_key]

                env.set_command(current_cmd)

            elif time.monotonic() - last_key_time > key_timeout_s:
                current_cmd = CMD_STAND.copy()

                if cmd_to_model is not None:
                    stand_key = tuple(current_cmd.astype(int))
                    if stand_key in cmd_to_model:
                        active_model, current_cmd = cmd_to_model[stand_key]

                env.set_command(current_cmd)

            if terminated or truncated:
                obs, _ = env.reset()
                env.set_command(current_cmd)
                obs[-CMD_DIM:] = current_cmd

    finally:
        env.close()
        cv2.destroyAllWindows()


def infer_algo_from_path(model_path):
    name = str(model_path).lower()

    if "sac" in name:
        return "sac"
    if "ppo" in name:
        return "ppo"

    sys.exit("Model filename must contain 'sac' or 'ppo' to determine algorithm.")


def run_inference(
    path: Path,
    algo: str = "sac",
    terrain_roughness: float | None = None,
    terrain_seed: int | None = None,
):
    if not Path(path).exists():
        sys.exit(f"Model path not found: {path}")

    if terrain_roughness is None:
        terrain_roughness = DEFAULT_TERRAIN_ROUGHNESS

    print(f"Loading {path} ...")
    model = ALGORITHMS[algo].load(str(path), device=ALGO_DEVICE[algo])

    print("\n=== Rough Inference (single model) ===")
    print("  W -> forward | A -> rotate left | D -> rotate right | ESC -> quit")
    print(f"  Roughness: {terrain_roughness}")
    print(f"  Terrain seed: {terrain_seed if terrain_seed is not None else 'random'}")

    try:
        _run_loop(
            model,
            CMD_STAND.copy(),
            cmd_to_model=None,
            window_name="Ant Rough",
            terrain_roughness=terrain_roughness,
            terrain_seed=terrain_seed,
        )
    except KeyboardInterrupt:
        print("\nRough inference stopped.")


def run_inference_multi_paths(
    model_map: dict[str, tuple[object, object]],
    terrain_roughness: float | None = None,
    terrain_seed: int | None = None,
):
    """Multi-model rough inference from explicit loaded model map.

    model_map format:
      {
        "stand": (loaded_model, CMD_STAND),
        "forward": (loaded_model, CMD_FORWARD),
      }
    """
    if not model_map:
        sys.exit("No models provided.")

    if terrain_roughness is None:
        terrain_roughness = DEFAULT_TERRAIN_ROUGHNESS

    cmd_to_model = {
        tuple(cmd.astype(int)): (model, cmd)
        for _, (model, cmd) in model_map.items()
    }

    if "stand" in model_map:
        active_model, current_cmd = model_map["stand"]
    else:
        active_model, current_cmd = next(iter(model_map.values()))

    print("\n=== Rough Inference (multi-model expert switching) ===")
    print("  W -> forward | A -> rotate left | D -> rotate right | ESC -> quit")
    print(f"  Loaded experts: {list(model_map.keys())}")
    print(f"  Roughness: {terrain_roughness}")
    print(f"  Terrain seed: {terrain_seed if terrain_seed is not None else 'random'}")

    try:
        _run_loop(
            active_model,
            current_cmd,
            cmd_to_model=cmd_to_model,
            window_name="Ant Rough",
            terrain_roughness=terrain_roughness,
            terrain_seed=terrain_seed,
        )
    except KeyboardInterrupt:
        print("\nRough inference stopped.")


def build_model_map_from_paths(model_paths: dict[str, Path], algo: str = "sac"):
    """Load models from explicit .zip paths and return a model_map for run_inference_multi_paths."""
    tag_to_cmd = {
        "stand": CMD_STAND,
        "forward": CMD_FORWARD,
        "left": CMD_LEFT,
        "right": CMD_RIGHT,
    }

    model_map: dict[str, tuple[object, object]] = {}
    loader = ALGORITHMS[algo]

    for tag, path in model_paths.items():
        if tag not in tag_to_cmd:
            print(f"  [warn] unknown tag '{tag}' - skipping")
            continue

        if not Path(path).exists():
            print(f"  [warn] path missing for '{tag}': {path}")
            continue

        print(f"  Loading {tag}: {path}")
        model_map[tag] = (
            loader.load(str(path), device=ALGO_DEVICE[algo]),
            tag_to_cmd[tag].copy(),
        )

    return model_map
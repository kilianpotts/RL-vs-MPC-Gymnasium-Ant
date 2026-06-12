"""
antpilot/config.py
Loads config.yaml and exposes typed constants for the rest of the package.
All magic numbers live here or in config.yaml — nowhere else.
"""

from pathlib import Path
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Load YAML
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_cfg = load_config()

# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

_art       = _cfg["artifacts"]
ARTIFACTS  = Path(_art["base"])
MODEL_DIR  = ARTIFACTS / _art["model_dir"]
CURVE_DIR  = ARTIFACTS / _art["curve_dir"]
TRAINING_CSV_DIR = ARTIFACTS / _art["training_csv_dir"]
LOG_FILE   = ARTIFACTS / _art["log_file"]
MODEL_STEM = _art["model_stem"]

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_cmd        = _cfg["commands"]
CMD_DIM     = int(_cmd["dim"])
CMD_STAND   = np.array(_cmd["stand"],   dtype=np.float32)
CMD_FORWARD = np.array(_cmd["forward"], dtype=np.float32)
CMD_LEFT    = np.array(_cmd["left"],    dtype=np.float32)
CMD_RIGHT   = np.array(_cmd["right"],   dtype=np.float32)

# Map string name -> array, used when parsing curriculum stages
CMD_MAP = {
    "stand":   CMD_STAND,
    "forward": CMD_FORWARD,
    "left":    CMD_LEFT,
    "right":   CMD_RIGHT,
}

CMD_NAMES = {
    (0, 0, 0): "STAND",
    (1, 0, 0): "FORWARD (W)",
    (0, 1, 0): "ROTATE LEFT (A)",
    (0, 0, 1): "ROTATE RIGHT (D)",
}

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

TRAINING = _cfg["training"]

# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

_algos      = _cfg["algorithms"]
ALGO_DEVICE = {name: vals.pop("device") for name, vals in _algos.items()}
ALGO_KWARGS = _algos

# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

_cur = _cfg["curriculum"]

CURRICULUM_MIN_SAMPLES = int(_cur["min_samples"])
CURRICULUM_MAX_SAMPLES = int(_cur["max_samples"])

# List of dicts: [{"name": "stand", "commands": {"stand": 1.0}}, ...]
CURRICULUM_STAGES: list[dict] = _cur["stages"]

def get_stage(name: str) -> dict:
    """Return the curriculum stage dict for the given name, or raise."""
    for stage in CURRICULUM_STAGES:
        if stage["name"] == name:
            return stage
    raise ValueError(f"Unknown curriculum stage '{name}'. "
                     f"Available: {[s['name'] for s in CURRICULUM_STAGES]}")
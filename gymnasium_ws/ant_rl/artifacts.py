"""
antpilot/artifacts.py
Save/load helpers for models, training curves, and the run log.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from .config import MODEL_DIR, CURVE_DIR, LOG_FILE, MODEL_STEM


def init_artifact_dirs():
    """Create artifact directories and log header if missing. Call once at startup."""
    for d in (MODEL_DIR, CURVE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.write_text("timestamp,model,tag,final_reward,auc,steps,mode\n")


def latest_checkpoint(algo: str, tag: str) -> Path | None:
    """Return path (without .zip) of the most recent checkpoint, or None."""
    prefix     = f"{MODEL_STEM}_{algo}_{tag}_"
    candidates = list(MODEL_DIR.glob(f"{prefix}*.zip"))
    if not candidates:
        return None
    return sorted(candidates)[-1].with_suffix("")  # strip .zip


def save_model(model, algo: str, tag: str, ts: str) -> Path:
    path = MODEL_DIR / f"{MODEL_STEM}_{algo}_{tag}_{ts}"
    model.save(str(path))
    print(f"  saved model -> {path}.zip")
    return path


def save_curve(ep_rewards: list, tag: str, ts: str) -> Path:
    path = CURVE_DIR / f"curve_{tag}_{ts}.png"
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(ep_rewards, alpha=0.4, color="steelblue", linewidth=0.8, label="ep reward")

    if len(ep_rewards) >= 20:
        kernel = np.ones(20) / 20
        rolled = np.convolve(ep_rewards, kernel, mode="valid")
        ax.plot(range(19, len(ep_rewards)), rolled,
                color="navy", linewidth=1.5, label="mean-20")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"Training curve - {tag} - {ts}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  saved curve  -> {path}")
    return path


def append_log(model_name: str, tag: str, ep_rewards: list, steps: int, mode: str):
    if not ep_rewards:
        return
    final_reward = float(np.mean(ep_rewards[-20:]))
    auc          = float(np.trapezoid(ep_rewards) if hasattr(np, "trapezoid") else np.trapz(ep_rewards))
    ts           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{model_name},{tag},{final_reward:.3f},{auc:.1f},{steps},{mode}\n")
    print(f"  logged       -> {LOG_FILE}")
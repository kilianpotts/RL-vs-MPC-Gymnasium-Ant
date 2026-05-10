#To DO
# -current stand is a bit querky think of good reward shaping (reward currently more of a stand up then a stand still)
# -implement curriculum learning building on stand
# -when in stage forward: most actions should be forward, but stand should be repeatedly sampled to avoid forgetting how to stand still
#- implement rotate rewards
#- consider symmetrie, maybe flip command and observation left-right and share model weights for rotate left and right? 


"""
Command-Conditioned Ant-v5
==========================
Command vector: [w, a, d]  — one-hot or zero vector
  [0, 0, 0]  ->  stand still  (no key)
  [1, 0, 0]  ->  forward      (W)
  [0, 1, 0]  ->  rotate left  (A)
  [0, 0, 1]  ->  rotate right (D)

Usage:
    python ant_cmd.py train --sac       # resume or start stage 1 (stand) with SAC
    python ant_cmd.py train --ppo       # resume or start stage 1 (stand) with PPO
  python ant_cmd.py train --retrain   # start fresh, keep old checkpoints
  python ant_cmd.py inference         # live keyboard control
"""

import os
import sys
import signal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# ---------------------------------------------------------------------------
# Artifact directories
# ---------------------------------------------------------------------------

ARTIFACTS = Path("/workspaces/gymnasium_ws/artifacts")
MODEL_DIR = ARTIFACTS / "models"
CURVE_DIR = ARTIFACTS / "curves"
LOG_FILE  = ARTIFACTS / "log.csv"

for d in (MODEL_DIR, CURVE_DIR):
    d.mkdir(parents=True, exist_ok=True)

if not LOG_FILE.exists():
    LOG_FILE.write_text("timestamp,model,tag,final_reward,auc,steps,mode\n")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CMD_DIM = 3

CMD_STAND   = np.array([0., 0., 0.], dtype=np.float32)
CMD_FORWARD = np.array([1., 0., 0.], dtype=np.float32)
CMD_LEFT    = np.array([0., 1., 0.], dtype=np.float32)
CMD_RIGHT   = np.array([0., 0., 1.], dtype=np.float32)

MODEL_STEM = "ant_cmd"

ALGORITHMS = {
    "sac": SAC,
    "ppo": PPO,
}

ALGO_KWARGS = {
    "sac": dict(
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
    ),
    "ppo": dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    ),
}

ALGO_DEVICE = {
    # SB3 recommends CPU for PPO with MLP policies.
    "sac": "auto",
    "ppo": "cpu",
}


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class CmdAnt(gym.Wrapper):
    """
    Appends a 3-dim command vector to the Ant-v5 observation and
    replaces the reward with a command-appropriate one.
    """

    def __init__(self, command: np.ndarray = CMD_STAND, render_mode=None):
        base = gym.make(
            "Ant-v5",
            render_mode=render_mode,
            exclude_current_positions_from_observation=False,
        )
        super().__init__(base)

        self.command = np.array(command, dtype=np.float32)

        lo = np.concatenate([self.env.observation_space.low,  np.zeros(CMD_DIM, np.float32)])
        hi = np.concatenate([self.env.observation_space.high, np.ones(CMD_DIM,  np.float32)])
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)

    def set_command(self, cmd: np.ndarray):
        self.command = np.array(cmd, dtype=np.float32)

    def _obs(self, raw):
        return np.concatenate([raw, self.command]).astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._obs(obs), info

    def step(self, action):
        obs, _r, terminated, truncated, info = self.env.step(action)
        reward = self._reward(obs, action, info)
        return self._obs(obs), reward, terminated, truncated, info

    def _reward(self, obs, action, info) -> float:
        w, a, d = self.command
        if w < .5 and a < .5 and d < .5:
            return self._r_stand(obs, action, info)
        elif w > .5:
            return self._r_forward(obs, action, info)
        elif a > .5:
            return self._r_rotate(obs, action, info, sign=+1)
        else:
            return self._r_rotate(obs, action, info, sign=-1)

    def _energy(self, action):
        return -0.001 * float(np.sum(action ** 2))

    def _r_stand(self, obs, action, info) -> float:
        torso_z  = float(obs[2])
        quat_w   = float(obs[3])
        quat_xyz = obs[4:7]
        x_vel    = float(info.get("x_velocity", 0.))
        y_vel    = float(info.get("y_velocity", 0.))
        joint_pos = obs[7:15]
        joint_pos = obs[7:15]
        hip_angles = joint_pos[[0, 2, 4, 6]]
        joint_vel = obs[21:29]

 
        # Exponential pull toward 0.75m — lying (~0.13m) -> ~0.0, standing (~0.75m) -> 5.0
        height_bon  =  5.0 * np.exp(-8.0 * max(0.0, 0.75 - torso_z))
 
        # Reward quat_w close to 1 (upright) — max 2.0 when perfectly level
        upright_bon =  2.0 * (quat_w ** 2)
 
        # Penalise tilt — quat_xyz near zero = upright, coefficient small to avoid freezing
        tilt_pen    = -0.5 * float(np.sum(quat_xyz ** 2))
 
        # Velocity penalty — only once off the ground
        vel_pen = -(x_vel ** 2 + y_vel ** 2) if torso_z > 0.3 else 0.0

        # Penalise asymetrical hips
        if torso_z > 0.7:
            hip_mean = np.mean(hip_angles)
            hip_dev = np.abs(hip_angles - hip_mean)
            tolerance = 0.35
            excess = np.maximum(0.0, hip_dev - tolerance)
            hip_sym_pen = -0.01 * float(np.mean(excess ** 2))
        else:
            hip_sym_pen = 0.0

        # Penalise moving legs
        joint_vel_pen = -0.005 * float(np.sum(joint_vel ** 2)) if torso_z > 0.7 else 0.0
    
        return height_bon + upright_bon + tilt_pen + vel_pen + hip_sym_pen + joint_vel_pen + self._energy(action)

    def _r_forward(self, obs, action, info) -> float:
        x_vel = float(info.get("x_velocity", 0.))
        return x_vel + self._energy(action)

    def _r_rotate(self, obs, action, info, sign: int) -> float:
        # Placeholder — yaw rate reward goes here in stage 3/4
        return self._energy(action)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(command, seed=0):
    def _init():
        e = CmdAnt(command=command)
        e.reset(seed=seed)
        return e
    return _init


def _latest_checkpoint(algo: str, tag: str) -> Path | None:
    prefix     = f"{MODEL_STEM}_{algo}_{tag}_"
    candidates = list(MODEL_DIR.glob(f"{prefix}*.zip"))
    if not candidates:
        return None
    return sorted(candidates)[-1].with_suffix("")  # strip .zip


def _save_model(model, algo: str, tag: str, ts: str) -> Path:
    path = MODEL_DIR / f"{MODEL_STEM}_{algo}_{tag}_{ts}"
    model.save(str(path))
    print(f"  saved model -> {path}.zip")
    return path


def _save_curve(ep_rewards: list, tag: str, ts: str) -> Path:
    path = CURVE_DIR / f"curve_{tag}_{ts}.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep_rewards, alpha=0.4, color="steelblue", linewidth=0.8, label="ep reward")

    if len(ep_rewards) >= 20:
        kernel = np.ones(20) / 20
        rolled = np.convolve(ep_rewards, kernel, mode="valid")
        ax.plot(range(19, len(ep_rewards)), rolled, color="navy", linewidth=1.5, label="mean-20")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"Training curve — {tag} — {ts}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  saved curve  -> {path}")
    return path


def _append_log(model_name: str, tag: str, ep_rewards: list, steps: int, mode: str):
    if not ep_rewards:
        return
    final_reward = float(np.mean(ep_rewards[-20:]))
    auc          = float(np.trapezoid(ep_rewards) if hasattr(np, "trapezoid") else np.trapz(ep_rewards))
    ts           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{model_name},{tag},{final_reward:.3f},{auc:.1f},{steps},{mode}\n")
    print(f"  logged       -> {LOG_FILE}")


# ---------------------------------------------------------------------------
# Callback — logging + early stopping on reward saturation
# ---------------------------------------------------------------------------

class TrainingCallback(BaseCallback):
    """
    Tracks episode rewards and stops early when reward has saturated.

    Saturation = the mean reward over the last `plateau_window` episodes
    improved by less than `plateau_threshold` vs the previous window.
    """

    def __init__(self, log_every=20, plateau_window=100, plateau_threshold=0.5):
        super().__init__()
        self.log_every         = log_every
        self.plateau_window    = plateau_window
        self.plateau_threshold = plateau_threshold
        self.ep_rewards: list[float] = []
        self._cur = 0.

    def _on_step(self) -> bool:
        self._cur += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self.ep_rewards.append(self._cur)
            self._cur = 0.
            n = len(self.ep_rewards)

            if n % self.log_every == 0:
                mean = np.mean(self.ep_rewards[-self.log_every:])
                print(f"  ep {n:4d} | mean reward = {mean:.2f}")

            # Early stop once we have two full windows to compare
            if n >= self.plateau_window * 2:
                old = np.mean(self.ep_rewards[-self.plateau_window*2 : -self.plateau_window])
                new = np.mean(self.ep_rewards[-self.plateau_window:])
                if abs(new - old) < self.plateau_threshold:
                    print(f"\nReward saturated (delta={new-old:.4f} over last "
                          f"{self.plateau_window} eps) — stopping early.")
                    return False  # signals SB3 to stop learn()

        return True


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    command   = CMD_STAND,
    timesteps = 500_000,
    n_envs    = 10,
    seed      = 42,
    tag       = "stand",
    retrain   = False,
    algo      = "sac",
):
    if algo not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algo}'. Available: {', '.join(ALGORITHMS)}")

    Algo = ALGORITHMS[algo]
    device = ALGO_DEVICE[algo]
    model_stem = f"{MODEL_STEM}_{algo}"

    env    = SubprocVecEnv([make_env(command, seed=seed+i) for i in range(n_envs)])
    latest = _latest_checkpoint(algo, tag)

    if latest and not retrain:
        print(f"Resuming from {latest}.zip ... (stem: {model_stem})")
        model = Algo.load(str(latest), env=env, device=device)
    else:
        if retrain and latest:
            print(f"--retrain: ignoring {latest}.zip — starting fresh.")
        else:
            print(f"New model — tag: {tag} | algo: {algo} | stem: {model_stem}")
        model = Algo(
            "MlpPolicy", env,
            verbose=1, seed=seed, device=device,
            **ALGO_KWARGS[algo],
        )

    callback    = TrainingCallback(log_every=20, plateau_window=100, plateau_threshold=0.5)
    interrupted = False

    def _sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\nCtrl+C — saving ...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint)

    try:
        model.learn(timesteps, callback=callback, log_interval=10)
        if not interrupted:
            print("\nTraining complete.")
    except KeyboardInterrupt:
        interrupted = True

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = _save_model(model, algo, tag, ts)
    _save_curve(callback.ep_rewards, tag, ts)
    _append_log(
        model_name = model_path.name,
        tag        = tag,
        ep_rewards = callback.ep_rewards,
        steps      = model.num_timesteps,
        mode       = "interrupted" if interrupted else ("retrain" if retrain else "resume"),
    )

    if interrupted:
        print("Progress saved. Re-run to resume.")

    try:
        env.close()
    except BrokenPipeError:
        pass  # subprocesses already dead from Ctrl+C — expected
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model_tag="stand"):
    try:
        import cv2
    except ImportError:
        sys.exit("pip install opencv-python")

    algo = "sac"
    for a in ALGORITHMS:
        if f"--{a}" in sys.argv:
            algo = a

    latest = _latest_checkpoint(algo, model_tag)
    if not latest:
        sys.exit(f"No saved model for tag '{model_tag}' with algo '{algo}' — train first.")

    print(f"Loading {latest}.zip ...")
    Algo = ALGORITHMS[algo]
    model = Algo.load(str(latest), device=ALGO_DEVICE[algo])

    env = CmdAnt(command=CMD_STAND, render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    current_cmd = CMD_STAND.copy()
    env.set_command(current_cmd)

    KEY_W   = ord('w')
    KEY_A   = ord('a')
    KEY_D   = ord('d')
    KEY_ESC = 27

    CMD_NAMES = {
        (0,0,0): "STAND",
        (1,0,0): "FORWARD (W)",
        (0,1,0): "ROTATE LEFT (A)",
        (0,0,1): "ROTATE RIGHT (D)",
    }

    print("\n=== Inference ===")
    print("  W -> forward | A -> rotate left | D -> rotate right | ESC -> quit\n")

    while True:
        obs[-CMD_DIM:] = current_cmd
        action, _      = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if frame is None:
            continue

        frame    = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cmd_name = CMD_NAMES.get(tuple(current_cmd.astype(int)), "?")
        cv2.putText(frame, f"CMD: {cmd_name}",      (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"reward: {reward:.3f}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Ant", frame)

        key = cv2.waitKey(16) & 0xFF
        if key == KEY_ESC:
            break
        elif key == KEY_W:
            current_cmd = CMD_FORWARD.copy()
        elif key == KEY_A:
            current_cmd = CMD_LEFT.copy()
        elif key == KEY_D:
            current_cmd = CMD_RIGHT.copy()
        else:
            current_cmd = CMD_STAND.copy()

        env.set_command(current_cmd)

        if terminated or truncated:
            obs, _ = env.reset()
            obs[-CMD_DIM:] = current_cmd

    env.close()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mode    = sys.argv[1] if len(sys.argv) > 1 else "train"
    retrain = "--retrain" in sys.argv
    algo    = "sac"

    for a in ALGORITHMS:
        if f"--{a}" in sys.argv:
            algo = a

    if mode == "train":
        train(command=CMD_STAND, timesteps=200_000, n_envs=10, tag="stand", retrain=retrain, algo=algo)
    elif mode == "inference":
        run_inference(model_tag="stand")
    else:
        print(__doc__)
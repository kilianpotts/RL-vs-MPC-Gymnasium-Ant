"""
antpilot/train.py
Training loop, env factory, and callback.
"""

import signal
import time
import numpy as np
from datetime import datetime

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from .env import CmdAnt
from .config import CMD_STAND, CMD_MAP, ALGO_KWARGS, ALGO_DEVICE, get_stage
from .artifacts import init_artifact_dirs, latest_checkpoint, save_model, save_curve, append_log

ALGORITHMS = {"sac": SAC, "ppo": PPO}

init_artifact_dirs()


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(stage_probs: dict[str, float], seed: int = 0):
    """
    stage_probs: {"stand": 0.2, "forward": 0.8}
    The initial command is sampled from stage_probs at env creation.
    """
    probs = dict(stage_probs)

    def _init():
        # Sample initial command from the stage distribution
        names  = list(probs.keys())
        total  = sum(probs.values())
        p      = [v / total for v in probs.values()]
        name   = np.random.choice(names, p=p)
        cmd    = CMD_MAP[name].copy()

        env = CmdAnt(command=cmd, stage_probs=probs)
        env.reset(seed=seed)
        return env

    return _init


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class TrainingCallback(BaseCallback):
    def __init__(self, log_every=20, plateau_window=None, plateau_threshold=0.5):
        super().__init__()
        self.log_every         = log_every
        self.plateau_window    = plateau_window
        self.plateau_threshold = plateau_threshold
        self.ep_rewards: list[float] = []
        self._cur_rewards: np.ndarray | None = None
        self._start_time = time.time()
        self._printed_csv_header = False
        self._next_log_episode = log_every
        self._last_logged_episode_count = 0

    def _print_csv_line(self, episodes: int, episode_reward: float, average_reward: float) -> None:
        elapsed = time.time() - self._start_time
        fps = self.model.num_timesteps / elapsed if elapsed > 0 else 0.0
        minutes, seconds = divmod(int(elapsed), 60)

        if not self._printed_csv_header:
            print("episodes,fps,time_elapsed,episode_reward,average_reward")
            self._printed_csv_header = True
        print(f"{episodes},{fps:.1f},{minutes}:{seconds:02d},{episode_reward:.2f},{average_reward:.2f}")
        self._last_logged_episode_count = episodes

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        dones = np.asarray(self.locals["dones"], dtype=bool)

        if self._cur_rewards is None:
            self._cur_rewards = np.zeros_like(rewards, dtype=np.float64)

        self._cur_rewards += rewards

        done_indices = np.flatnonzero(dones)
        for idx in done_indices:
            self.ep_rewards.append(float(self._cur_rewards[idx]))
            self._cur_rewards[idx] = 0.0

        n = len(self.ep_rewards)

        while n >= self._next_log_episode:
            episode_reward = self.ep_rewards[self._next_log_episode - 1]
            avg_reward = np.mean(self.ep_rewards[self._next_log_episode - self.log_every:self._next_log_episode])
            self._print_csv_line(self._next_log_episode, episode_reward, float(avg_reward))
            self._next_log_episode += self.log_every

        if self.plateau_window is not None and n >= self.plateau_window * 2:
            old = np.mean(self.ep_rewards[-self.plateau_window * 2 : -self.plateau_window])
            new = np.mean(self.ep_rewards[-self.plateau_window:])
            if abs(new - old) < self.plateau_threshold:
                print(f"\nReward saturated (delta={new - old:.4f} over last "
                        f"{self.plateau_window} eps) — stopping early.")
                return False

        return True

    def _on_training_end(self) -> None:
        n = len(self.ep_rewards)
        if n == self._last_logged_episode_count:
            return

        if n == 0:
            self._print_csv_line(0, 0.0, 0.0)
            return

        window = min(self.log_every, n)
        episode_reward = self.ep_rewards[-1]
        avg_reward = float(np.mean(self.ep_rewards[-window:]))
        self._print_csv_line(n, float(episode_reward), avg_reward)


# ---------------------------------------------------------------------------
# Train (unified for both expert and curriculum modes)
# ---------------------------------------------------------------------------

def train(
    command: str = None,            # For expert mode: command name (stand, forward, left, right)
    stage: str = None,              # For curriculum mode: stage name from config.yaml
    timesteps: int = 20_000,
    n_envs: int = 10,
    seed: int = 42,
    retrain: bool = False,
    algo: str = "sac",
):
    """
    Train a policy (expert or curriculum mode).
    
    Expert mode (single command, 100% probability):
        train(command="stand", algo="sac", retrain=False)
        
    Curriculum mode (multiple commands with stage-defined probabilities):
        train(stage="forward", algo="ppo", retrain=False)
    
    Args:
        command: Command name for expert mode (stand, forward, left, right)
        stage: Curriculum stage name for curriculum mode
        timesteps: Number of training timesteps
        n_envs: Number of parallel environments
        seed: Random seed
        retrain: If True, start fresh and ignore existing checkpoints
        algo: Algorithm (sac or ppo)
    """
    if algo not in ALGORITHMS:
        raise ValueError(f"Unknown algo '{algo}'. Available: {', '.join(ALGORITHMS)}")
    
    # Determine mode and get stage_probs and tag
    if command is not None:
        if command not in CMD_MAP:
            raise ValueError(f"Unknown command '{command}'. Available: {', '.join(CMD_MAP.keys())}")
        stage_probs = {command: 1.0}
        tag = command
        mode_label = f"expert ({command})"
    elif stage is not None:
        stage_cfg = get_stage(stage)
        stage_probs = stage_cfg["commands"]
        tag = stage_cfg["name"]
        mode_label = f"curriculum ({stage})"
    else:
        raise ValueError("Must provide either 'command' (expert mode) or 'stage' (curriculum mode)")
    
    Algo = ALGORITHMS[algo]
    device = ALGO_DEVICE[algo]
    
    env = SubprocVecEnv([make_env(stage_probs, seed=seed + i) for i in range(n_envs)])
    latest = latest_checkpoint(algo, tag)
    
    if latest and not retrain:
        print(f"Resuming from {latest}.zip ...")
        model = Algo.load(str(latest), env=env, device=device)
    else:
        if retrain and latest:
            print(f"--retrain: ignoring {latest}.zip — starting fresh.")
        else:
            print(f"New model — {mode_label} | algo: {algo}")
        model = Algo(
            "MlpPolicy", env,
            verbose=0, seed=seed, device=device,
            **ALGO_KWARGS[algo],
        )
    
    callback = TrainingCallback(
        log_every=20,
        plateau_window=None,
        plateau_threshold=0.5,
    )
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
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = save_model(model, algo, tag, ts)
    save_curve(callback.ep_rewards, tag, ts)
    append_log(
        model_name=model_path.name,
        tag=tag,
        ep_rewards=callback.ep_rewards,
        steps=model.num_timesteps,
        mode="interrupted" if interrupted else ("retrain" if retrain else "resume"),
    )
    
    if interrupted:
        print("Progress saved. Re-run to resume.")
    
    try:
        env.close()
    except (BrokenPipeError, EOFError, ConnectionResetError):
        pass
    
    return model


# Convenience aliases for backwards compatibility
def train_expert(command: str = "stand", **kwargs):
    """Train expert policy for a single command."""
    return train(command=command, **kwargs)


def train_curriculum(stage: str = "stand", **kwargs):
    """Train curriculum stage with multiple commands."""
    return train(stage=stage, **kwargs)
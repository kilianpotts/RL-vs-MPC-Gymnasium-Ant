"""
antpilot/env.py
CmdAnt — Ant-v5 wrapper that appends a command vector to observations
and dispatches to command-specific reward functions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import (
    CMD_DIM, CMD_STAND, CMD_FORWARD, CMD_LEFT, CMD_RIGHT, CMD_MAP,
    CURRICULUM_MIN_SAMPLES, CURRICULUM_MAX_SAMPLES,
)


class CmdAnt(gym.Wrapper):
    """
    Wraps Ant-v5 so that:
      obs    = [ant_obs | command]   (ant_obs_dim + CMD_DIM)
      reward = command-specific function

    Command vector layout:
      [0, 0, 0]  ->  stand still  (no key)
      [1, 0, 0]  ->  forward      (W)
      [0, 1, 0]  ->  rotate left  (A)
      [0, 0, 1]  ->  rotate right (D)

    """

    def __init__(
        self,
        command: np.ndarray = None,
        stage_probs: dict[str, float] = None,
        render_mode: str = None,
    ):

        base = gym.make(
            "Ant-v5",
            render_mode=render_mode,
            exclude_current_positions_from_observation=False,
        )
        super().__init__(base)

        self.command       = np.array(command, dtype=np.float32)
        
        # Command scheduling probabilities
        if stage_probs is None:
            stage_probs = {}
        self._stage_probs = stage_probs
        self._stage_names = list(stage_probs.keys())

        # Scheduling state
        self._steps_held      = 0
        self._hold_for        = self._sample_hold_duration()

        # Extend observation space
        low_base = self.env.observation_space.low.astype(np.float32, copy=False)
        high_base = self.env.observation_space.high.astype(np.float32, copy=False)
        lo = np.concatenate([low_base, np.zeros(CMD_DIM, dtype=np.float32)])
        hi = np.concatenate([high_base, np.ones(CMD_DIM, dtype=np.float32)])
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)



    def _sample_hold_duration(self) -> int:
        return np.random.randint(CURRICULUM_MIN_SAMPLES, CURRICULUM_MAX_SAMPLES + 1)

    def _sample_command(self) -> np.ndarray:
        if not self._stage_names:
            return self.command.copy()
        # Convert dict to probability array in correct order
        probs = [self._stage_probs[name] for name in self._stage_names]
        name = np.random.choice(self._stage_names, p=probs)
        return CMD_MAP[name].copy()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def set_command(self, cmd: np.ndarray):
        """Manually override command (used during inference)."""
        self.command = np.array(cmd, dtype=np.float32)

    def _obs(self, raw: np.ndarray) -> np.ndarray:
        return np.concatenate([raw, self.command]).astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._obs(obs), info

    def step(self, action):
        obs, _r, terminated, truncated, info = self.env.step(action)

        # scheduling — sample new command after hold_for steps
        if self._stage_names:
            self._steps_held += 1
            if self._steps_held >= self._hold_for:
                self.command     = self._sample_command()
                self._steps_held = 0
                self._hold_for   = self._sample_hold_duration()

        reward = self._reward(obs, action, info)
        return self._obs(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward dispatch
    # ------------------------------------------------------------------

    def _reward(self, obs, action, info) -> float:
        w, a, d = self.command
        if   w < .5 and a < .5 and d < .5: 
            return self._r_stand(obs, action, info)
        elif w > .5:                        
            return self._r_forward(obs, action, info)
        elif a > .5:                        
            return self._r_rotate(obs, action, info, sign=+1)
        else:                               
            return self._r_rotate(obs, action, info, sign=-1)

    def _energy(self, action) -> float:
        return -0.001 * float(np.sum(action ** 2))

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _r_stand(self, obs, action, info) -> float:
        torso_z    = float(obs[2])
        quat_w     = float(obs[3])
        quat_xyz   = obs[4:7]
        x_vel      = float(info.get("x_velocity", 0.))
        y_vel      = float(info.get("y_velocity", 0.))
        joint_pos  = obs[7:15]
        hip_angles = joint_pos[[0, 2, 4, 6]]
        joint_vel  = obs[21:29]

        height_bon  =  5.0 * np.exp(-8.0 * max(0.0, 0.75 - torso_z))
        upright_bon =  2.0 * (quat_w ** 2)
        tilt_pen    = -0.5 * float(np.sum(quat_xyz ** 2))
        vel_pen     = -(x_vel ** 2 + y_vel ** 2) if torso_z > 0.3 else 0.0

        if torso_z > 0.7:
            hip_mean    = np.mean(hip_angles)
            excess      = np.maximum(0.0, np.abs(hip_angles - hip_mean) - 0.35)
            hip_sym_pen = -0.01 * float(np.mean(excess ** 2))
            jv_pen      = -0.005 * float(np.sum(joint_vel ** 2))
        else:
            hip_sym_pen = 0.0
            jv_pen      = 0.0

        return height_bon + upright_bon + tilt_pen + vel_pen + hip_sym_pen + jv_pen + self._energy(action)

    def _r_forward(self, obs, action, info) -> float:
        torso_z = float(obs[2])
        quat_w  = float(obs[3])
        x_vel   = float(info.get("x_velocity", 0.0))
        y_vel   = float(info.get("y_velocity", 0.0))

        posture_score = float(np.clip((torso_z - 0.45) / 0.30, 0.0, 1.0))
        upright_score = float(np.clip(quat_w * quat_w, 0.0, 1.0))
        forward_term  = float(np.clip(x_vel, -1.0, 2.0))
        lateral_pen   = -0.2  * (y_vel ** 2)
        stillness_pen = -1.5  * float(np.exp(-8.0 * (x_vel**2 + y_vel**2)))

        return (
            0.8
            + 1.1 * forward_term
            + 0.4 * posture_score
            + 0.4 * upright_score
            + lateral_pen
            + stillness_pen
            + self._energy(action)
        )

    def _r_rotate(self, obs, action, info, sign: int) -> float:
        torso_z = float(obs[2])
        quat_w = float(obs[3])

        x_vel = float(info.get("x_velocity", 0.0))
        y_vel = float(info.get("y_velocity", 0.0))

        yaw_rate = float(obs[20])


#        if torso_z < 0.35:
#            return -5.0 + self._energy(action)

        posture_score = float(np.clip((torso_z - 0.45) / 0.30, 0.0, 1.0))
        upright_score = float(np.clip(quat_w * quat_w, 0.0, 1.0))

        signed_yaw_rate = sign * yaw_rate
        rotate_term = float(np.clip(signed_yaw_rate, -1.0, 2.0))

        translation_pen = -0.4 * (x_vel ** 2 + y_vel ** 2)
        wrong_dir_pen = -0.5 * max(0.0, -signed_yaw_rate)
        stillness_pen = -1.0 * float(np.exp(-6.0 * (yaw_rate ** 2)))

        return (
            0.5
            + 1.2 * rotate_term
            + 0.4 * posture_score
            + 0.4 * upright_score
            + translation_pen
            + wrong_dir_pen
            + stillness_pen
            + self._energy(action)
        )
    
    
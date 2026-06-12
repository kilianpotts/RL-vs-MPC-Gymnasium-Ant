"""
antpilot/env.py
CmdAnt — Ant-v5 wrapper that appends a command vector to observations
and dispatches to command-specific reward functions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

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
        terrain: str = "flat",
        terrain_roughness: float = 0.0,
    ):

        xml_file = os.path.join(
            os.path.dirname(__file__),
            "assets",
            "ant.xml",
        )

        base = gym.make(
            "Ant-v5",
            xml_file=xml_file,
            render_mode=render_mode,
            exclude_current_positions_from_observation=False,
        )
        super().__init__(base)

        self.terrain = str(terrain).lower()
        self.terrain_roughness = float(terrain_roughness)

        if self.terrain not in ("flat", "rough"):
            raise ValueError(f"Unknown terrain '{terrain}'. Use 'flat' or 'rough'.")

        if command is None:
            command = CMD_STAND

        self.command = np.array(command, dtype=np.float32)
        
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

    def _hfield_slice(self):
        model = self.env.unwrapped.model

        try:
            hfield_id = model.hfield("terrain").id
        except Exception:
            return None

        start = model.hfield_adr[hfield_id]
        nrow = model.hfield_nrow[hfield_id]
        ncol = model.hfield_ncol[hfield_id]
        size = nrow * ncol

        return model, start, size, nrow, ncol


    def _clear_terrain(self):
        data = self._hfield_slice()
        if data is None:
            return

        model, start, size, _nrow, _ncol = data
        model.hfield_data[start:start + size] = 0.0


    def _smooth_heightfield(self, terrain: np.ndarray, passes: int = 4) -> np.ndarray:
        for _ in range(passes):
            terrain = (
                terrain
                + np.roll(terrain, 1, axis=0)
                + np.roll(terrain, -1, axis=0)
                + np.roll(terrain, 1, axis=1)
                + np.roll(terrain, -1, axis=1)
            ) / 5.0
        return terrain


    def _generate_terrain(self):
        data = self._hfield_slice()
        if data is None:
            return

        model, start, size, nrow, ncol = data

        terrain = self.env.unwrapped.np_random.normal(
            loc=0.0,
            scale=1.0,
            size=(nrow, ncol),
        )

        terrain = self._smooth_heightfield(terrain, passes=4)

        # Startbereich in der Mitte etwas flacher halten
        c0, c1 = nrow // 2, ncol // 2
        flat_radius = 5
        terrain[c0-flat_radius:c0+flat_radius, c1-flat_radius:c1+flat_radius] *= 0.10

        # Auf 0..1 normalisieren
        terrain -= terrain.min()
        max_h = terrain.max()
        if max_h > 1e-8:
            terrain /= max_h

        # terrain_roughness steuert, wie viel der XML-Höhe genutzt wird.
        # Beispiel bei XML size-Höhe 0.20:
        # 0.25 -> ca. 0.05 maximale Höhe
        # 0.50 -> ca. 0.10 maximale Höhe
        # 1.00 -> ca. 0.20 maximale Höhe
        terrain *= self.terrain_roughness

        model.hfield_data[start:start + size] = terrain.ravel()

    def _sample_hold_duration(self) -> int:
        return np.random.randint(CURRICULUM_MIN_SAMPLES, CURRICULUM_MAX_SAMPLES + 1)

    def _sample_command(self) -> np.ndarray:
        if not self._stage_names:
            return self.command.copy()

        weights = np.array(
            [self._stage_probs[name] for name in self._stage_names],
            dtype=np.float64,
        )
        weights = weights / weights.sum()

        name = np.random.choice(self._stage_names, p=weights)
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

        if self.terrain == "rough":
            self._generate_terrain()
        else:
            self._clear_terrain()

        return self._obs(obs), info

    def step(self, action):
        obs, _r, terminated, truncated, info = self.env.step(action)

        # Reward für den Command berechnen, der auch in der vorherigen Observation stand.
        reward = self._reward(obs, action, info)

        # Command für den nächsten Schritt wechseln.
        if self._stage_names:
            self._steps_held += 1
            if self._steps_held >= self._hold_for:
                self.command = self._sample_command()
                self._steps_held = 0
                self._hold_for = self._sample_hold_duration()

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
        return -0.005 * float(np.sum(action ** 2))
    

    def _upright(self, obs) -> float:
        qw, qx, qy, qz = obs[3:7]
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        return float(np.clip((up_z - 0.4) / 0.6, 0.0, 1.0))

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
        roll_pitch_yaw_vel = obs[18:21]

        height_bon  =  5.0 * np.exp(-8.0 * max(0.0, 0.75 - torso_z))
        upright_bon =  2.0 * (quat_w ** 2)
        tilt_pen    = -0.5 * float(np.sum(quat_xyz ** 2))
        vel_pen     = -(x_vel ** 2 + y_vel ** 2) if torso_z > 0.3 else 0.0
        ang_vel_pen = -0.5 * float(np.sum(obs[18:21] ** 2))

        if torso_z < 0.35:
            return -5.0 + self._energy(action)

        if torso_z > 0.7:
            hip_mean    = np.mean(hip_angles)
            excess      = np.maximum(0.0, np.abs(hip_angles - hip_mean) - 0.35)
            hip_sym_pen = -0.01 * float(np.mean(excess ** 2))
            jv_pen      = -0.02 * float(np.sum(joint_vel ** 2))
        else:
            hip_sym_pen = 0.0
            jv_pen      = 0.0

        return height_bon + upright_bon + tilt_pen + vel_pen + hip_sym_pen + jv_pen + ang_vel_pen + self._energy(action)

    def _r_forward(self, obs, action, info) -> float:
        torso_z = float(obs[2])

        # if torso_z < 0.35:
        #     return -1.5 + self._energy(action)

        x_vel = float(info.get("x_velocity", 0.0))
        y_vel = float(info.get("y_velocity", 0.0))

        #project x_vel and y_vel into ant frame
        qw, qx, qy, qz = obs[3:7]
        # yaw from quaternion
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        forward_vel =  np.cos(yaw)*x_vel + np.sin(yaw)*y_vel
        lateral_vel = -np.sin(yaw)*x_vel + np.cos(yaw)*y_vel

        upright = self._upright(obs)

        # ----------------------------
        # locomotion objective
        # ----------------------------

        # Prevent negative values from entering tanh to keep the gradient clean
        clipped_forward_vel = max(0.0, forward_vel)
        
        # alpha=0.75 ensures the "knee" of the saturation curve is at 1.5 m/s
        # Scale weight (e.g., 5.0) determines the maximum possible reward tokens
        forward_term = 5.0 * float(np.tanh(0.75 * clipped_forward_vel))

        # ----------------------------
        # smoothness terms
        # ----------------------------

        z_vel = float(obs[17])

        ang_vel = obs[18:21]
        joint_vel = obs[21:29]

        #penalize bouncing more, to avoid hopping instead of smooth gait
        bounce_pen = -1.5 * (z_vel ** 2)

        ang_vel_pen = -0.05 * float(np.sum(ang_vel ** 2))

        joint_vel_pen = -0.01 * float(np.sum(joint_vel ** 2))

        # discourage sideways drift
        lateral_pen = -0.3 * (lateral_vel ** 2)
        backward_pen = -1.0 * max(0.0, -forward_vel)


        return (
            forward_term * upright
            + 0.8 * upright
            + bounce_pen
            + ang_vel_pen
            + joint_vel_pen
            + lateral_pen
            + backward_pen
            + self._energy(action)
        )
    def _r_rotate(self, obs, action, info, sign: int) -> float:
        torso_z = float(obs[2])

        x_vel = float(info.get("x_velocity", 0.0))
        y_vel = float(info.get("y_velocity", 0.0))

        yaw_rate = float(obs[20])

        upright_score = self._upright(obs)  
        signed_yaw_rate = sign * yaw_rate

        yaw_progress = float(np.clip(signed_yaw_rate, -1.0, 2.0))
        target_yaw_rate = 1.2
        yaw_error = signed_yaw_rate - target_yaw_rate
        target_bonus = float(np.exp(-1.5 * (yaw_error ** 2))) * upright_score

        translation_pen = -0.4 * (x_vel ** 2 + y_vel ** 2)
        wrong_dir_pen = -0.5 * max(0.0, -signed_yaw_rate)
        stillness_pen = -2.0 * float(np.exp(-6.0 * (yaw_rate ** 2)))

        return (
            2 * yaw_progress
            + 2 * target_bonus
            + 0.7 * upright_score
            + translation_pen
            + wrong_dir_pen
            + stillness_pen
            + self._energy(action)
        )
    
    
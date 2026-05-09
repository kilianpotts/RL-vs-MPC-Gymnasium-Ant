import gymnasium as gym
import numpy as np


class StandingAntWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, original_reward, terminated, trunacted, info = self.env.step(action)

        torso_height = obs[0]
        velocities = obs[13:]

        if 0.2 <= torso_height <= 1.0:
            alive_reward = 1.0
        else:
            alive_reward = -1.0
        
        velocity_penalty = 0.01 * np.sum(np.square(velocities))

        action_penalty = 0.05 * np.sum(np.square(action))

        reward = alive_reward - velocity_penalty - action_penalty

        return obs, reward, terminated, trunacted, info
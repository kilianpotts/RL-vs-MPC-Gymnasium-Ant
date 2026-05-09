import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from standing_ant_wrapper import StandingAntWrapper

def make_standing_ant_env():
    env = gym.make("Ant-v5", render_mode=None)
    env = StandingAntWrapper(env)
    return env
 
# damit jeder Prozess nur einmal ausgeführt wird
if __name__ == "__main__":
    # Parallel environments
    env = make_vec_env(
        make_standing_ant_env,
        n_envs=4, # 4 ant enviroments
        vec_env_cls=SubprocVecEnv,
    )

    # PPO Policy erstellen
    model = PPO(
        "MlpPolicy", # Multi-Layer Perceptron
        env,
        verbose=1, # output data in console
        device="cpu",
        learning_rate=3e-4, #default
        n_steps=2048, # default
        batch_size=64, # default
        n_epochs=10, # default
        gamma=0.99, # default
        gae_lambda=0.95, #default
        clip_range=0.2, #default
        ent_coef=0.0, #default
    )

    # Trainieren
    model.learn(total_timesteps=1_000_000)

    # Modell speichern
    model.save("ppo_ant")

    env.close()
import gymnasium as gym
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO


os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)


env = gym.make("Ant-v5", render_mode="rgb_array")

model = PPO.load("ppo_ant")


num_episodes = 20
max_steps_per_episode = 1000

episode_rewards = []


for episode in range(num_episodes):

    obs, info = env.reset()

    total_reward = 0.0
    stop_visualization = False

    for t in range(max_steps_per_episode):

        # entspricht: action, _states = model.predict(obs)
        action, _states = model.predict(obs, deterministic=True)

        # entspricht: obs, rewards, dones, info = env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

        # entspricht: vec_env.render("human")
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("PPO Ant", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            stop_visualization = True
            break

        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.2f}")

    if stop_visualization:
        break


env.close()
cv2.destroyAllWindows()


# Rewards speichern
df = pd.DataFrame({
    "episode": np.arange(1, len(episode_rewards) + 1),
    "reward": episode_rewards
})

df.to_csv("logs/ppo_policy_rewards.csv", index=False)


# Rewards plotten
plt.figure()
plt.plot(df["episode"], df["reward"], marker="o")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Policy - Ant")
plt.grid(True)
plt.savefig("plots/ppo_policy_rewards.png")
plt.show()
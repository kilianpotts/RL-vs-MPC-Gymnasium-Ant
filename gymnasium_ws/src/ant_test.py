import gymnasium as gym
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os




# IMPORTANT: use rgb_array instead of human
env = gym.make("Ant-v5", render_mode="rgb_array")


def random_policy(obs, action_space):
    return action_space.sample()

def zero_policy(obs, action_space):
    return np.zeros(action_space.shape, dtype=action_space.dtype)

def constant_policy(obs, action_space):
    action = np.full(action_space.shape, 0.2, dtype=action_space.dtype)
    return action

def sinus_policy(obs, t, action_space):
    action = np.zeros(action_space.shape, dtype=np.float32)
    action[0] = np.sin(0.1 * t)
    action[1] = -np.sin(0.1 * t)
    action[2] = np.sin(0.1 * t)
    action[3] = -np.sin(0.1 * t)
    return action.astype(action_space.dtype)

num_episodes = 20
max_steps_per_episode = 1000

episode_rewards = []


for episode in range(num_episodes):

    obs, info = env.reset()

    total_reward = 0
    done = False

    for _ in range(max_steps_per_episode):
        action = random_policy(obs, env.action_space)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

        # Get rendered frame (H, W, 3) RGB
        frame = env.render()

        # Convert RGB → BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Show frame
        cv2.imshow("MuJoCo Ant", frame)

        # Needed for window to update
        if cv2.waitKey(1) & 0xFF == 27:  # press ESC to quit
            break

        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.2f}")

env.close()
cv2.destroyAllWindows()


# safe Rewards
df = pd.DataFrame({
    "episode": np.arange(1, len(episode_rewards) + 1),
    "reward": episode_rewards
})

df.to_csv("gymnasium_ws/logs/random_policy_rewords.csv", index=False)

# plotting rewards
plt.figure()
plt.plot(df["episode"], df["reward"], marker="o")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Random Policy - Ant")
plt.grid(True)
plt.savefig("gymnasium_ws/plots/random_policy_rewards.png")
plt.show()
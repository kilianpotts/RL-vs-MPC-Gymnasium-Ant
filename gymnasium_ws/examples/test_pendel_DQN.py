import gymnasium as gym
import numpy as np
import time
from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim

# ===============================================================
# 1) Q-Netz, Replay Buffer, Policy
# ===============================================================
class QNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s   = torch.as_tensor(np.array([t.s   for t in batch]), dtype=torch.float32)
        a   = torch.as_tensor(np.array([t.a   for t in batch]), dtype=torch.int64).unsqueeze(1)
        r   = torch.as_tensor(np.array([t.r   for t in batch]), dtype=torch.float32).unsqueeze(1)
        s2  = torch.as_tensor(np.array([t.s2  for t in batch]), dtype=torch.float32)
        done= torch.as_tensor(np.array([t.done for t in batch]), dtype=torch.float32).unsqueeze(1)
        return s, a, r, s2, done
    def __len__(self):
        return len(self.buf)

def epsilon_greedy(qnet, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q = qnet(s)                    # shape [1, n_actions]
        return int(torch.argmax(q, dim=1).item())

# ===============================================================
# 2) Umgebung & Netze
# ===============================================================
env = gym.make("CartPole-v1", render_mode=None)  # Training ohne Rendern
state_dim = env.observation_space.shape[0]       # 4
n_actions = env.action_space.n                   # 2

qnet      = QNet(state_dim, n_actions)
targetnet = QNet(state_dim, n_actions)
targetnet.load_state_dict(qnet.state_dict())     # initial sync
targetnet.eval()

buffer = ReplayBuffer(capacity=50_000)
optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
loss_fn = nn.SmoothL1Loss()  # Huber-Loss

# Hyperparameter
gamma = 0.99
batch_size = 64
train_episodes = 400
max_steps = 500
start_learning_after = 1_000      # warmup transitions
target_update_every  = 200        # steps
epsilon_start, epsilon_end = 1.0, 0.05
epsilon_decay_episodes = 300

def epsilon_by_episode(ep):
    if ep >= epsilon_decay_episodes:
        return epsilon_end
    return epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_episodes)

global_step = 0
reward_history = deque(maxlen=50)

# ===============================================================
# 3) Training
# ===============================================================
print("=== Training DQN Agent (CartPole) ===")
for ep in range(train_episodes):
    s, info = env.reset()
    ep_reward = 0.0
    epsilon = epsilon_by_episode(ep)

    for t in range(max_steps):
        a = epsilon_greedy(qnet, s, epsilon, n_actions)
        s2, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        buffer.push(s, a, r, s2, done)
        s = s2
        ep_reward += r
        global_step += 1

        # Lernen erst nach Warmup und wenn genug Samples da sind
        if len(buffer) >= max(batch_size, start_learning_after):
            # Sample Batch
            bs, ba, br, bs2, bdone = buffer.sample(batch_size)

            # Q(s,a) aus aktuellem Netz
            q_sa = qnet(bs).gather(1, ba)  # shape [B,1]

            # Target: r + gamma * max_a' Q_target(s', a') * (1-done)
            with torch.no_grad():
                q_next = targetnet(bs2).max(dim=1, keepdim=True).values
                td_target = br + gamma * q_next * (1.0 - bdone)

            # Loss & Update
            loss = loss_fn(q_sa, td_target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
            optimizer.step()

            # Target-Netz regelmäßig synchronisieren
            if global_step % target_update_every == 0:
                targetnet.load_state_dict(qnet.state_dict())

        if done:
            break

    reward_history.append(ep_reward)
    if (ep + 1) % 20 == 0:
        avg_last = np.mean(reward_history) if reward_history else 0.0
        print(f"Episode {ep+1:3d}/{train_episodes} | Reward: {ep_reward:5.1f} | "
              f"Avg(50): {avg_last:5.1f} | epsilon={epsilon:.3f} | steps={global_step}")

env.close()


# ===============================================================
# 4) Test: Ausführen der gelernten Policy (mit Rendern)
# ===============================================================
print("\n=== Test der gelernten DQN-Policy (gerendert) ===")
import gymnasium as gym
import numpy as np
import time
import os


# Create directory for saving videos
dqn_video_dir = "./dqn_videos"
os.makedirs(dqn_video_dir, exist_ok=True)


env = gym.make("CartPole-v1", render_mode="rgb_array") # Change render_mode to "human" when you run the code locally
env = gym.wrappers.RecordVideo(env, dqn_video_dir, episode_trigger=lambda x: True, name_prefix="dqn_episode") # Record all episodes

s, info = env.reset(seed=42) # Added seed for reproducibility
done = False
total_reward = 0.0


while not done:
    # Need to ensure qnet and n_actions are available from the training cell
    # If running this cell independently, you might need to re-define/load the qnet
    st = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0) # Assuming qnet is on CPU
    # If qnet is on GPU, uncomment the line below and remove the one above
    # st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = qnet(st)
        a = torch.argmax(q_values, dim=1).item()  # reine Exploitation

    s, r, terminated, truncated, info = env.step(a)
    total_reward += r
    done = terminated or truncated

print(f"Test beendet. Total reward: {total_reward:.1f}")
env.close()

import glob
import io
from base64 import b64encode
from IPython.display import HTML, display

# Find the first recorded video file in the dqn directory
video_files = glob.glob("./dqn_videos/*.mp4")
if video_files:
    video_path = video_files[0]
    print(f"Displaying video from: {video_path}")

    # Function to display video in Colab
    def show_video(video_path):
        mp4 = open(video_path, 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        return HTML("""
        <video width=400 controls>
            <source src="%s" type="video/mp4">
        </video>
        """ % data_url)

    display(show_video(video_path))
else:
    print("No video files found in ./dqn_videos.")
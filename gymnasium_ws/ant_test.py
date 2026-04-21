import gymnasium as gym
import cv2

# IMPORTANT: use rgb_array instead of human
env = gym.make("Ant-v5", render_mode="rgb_array")

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    # Get rendered frame (H, W, 3) RGB
    frame = env.render()

    # Convert RGB → BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Show frame
    cv2.imshow("MuJoCo Ant", frame)

    # Needed for window to update
    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to quit
        break

    if terminated or truncated:
        obs, info = env.reset()

env.close()
cv2.destroyAllWindows()
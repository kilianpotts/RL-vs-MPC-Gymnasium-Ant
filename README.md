# Gymnasium Ant: RL vs MPC


## Development Container Setup

### GPU Rendering with EGL

This dev container uses **EGL** for headless GPU rendering via the environment variable `MUJOCO_GL=egl`. This approach avoids compatibility issues with GLFW in Docker and WSL environments.

**Why EGL?**
- Direct GPU rendering without a display server
- Eliminates GLFW/Docker/WSL integration challenges
- Frames are captured and rendered via OpenCV

**Current Configuration**
- Tested and working with WSL + NVIDIA GPU
- `MUJOCO_GL=egl` is set in postCreateCommand (devcontainer.json)
- Frame output handled through OpenCV pipelines

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (OpenGL + X11 + Qt + MuJoCo + OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    sudo \
    nano \
    git \
    ffmpeg \
    # OpenGL / rendering
    libgl1 \
    libosmesa6 \
    libglfw3 \
    libegl1 \
    libgles2 \
    # X11 / Qt / OpenCV GUI support
    libglib2.0-0 \
    libgtk-3-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libxrandr2 \
    libxi6 \
    libxcb1 \
    libx11-xcb1 \
    libfontconfig1 \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install GPU-enabled PyTorch 
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# RL + MuJoCo stack
RUN pip3 install --no-cache-dir \
    "gymnasium[classic-control,mujoco]>=0.29" \
    mujoco \
    "stable-baselines3>=2.3.0" \
    opencv-python

# Analysis + plotting + trajectory tooling
RUN pip3 install --no-cache-dir \
    matplotlib \
    pandas \
    seaborn \
    scipy \
    pyyaml \
    imageio \
    tqdm \

# (Optional but helpful for MuJoCo rendering)
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Create non-root user
RUN useradd -ms /bin/bash devuser && \
    usermod -aG sudo devuser && \
    echo 'devuser ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/devuser && \
    chmod 0440 /etc/sudoers.d/devuser && \
    mkdir -p /workspaces/gymnasium_ws && \
    chown -R devuser:devuser /workspaces

USER devuser
WORKDIR /workspaces/gymnasium_ws

CMD ["bash"]
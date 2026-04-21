FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (OpenGL + X11 + Qt + MuJoCo + OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    sudo \
    nano \
    git \
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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip + install Python packages
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        "gymnasium[classic-control,mujoco]" \
        opencv-python

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
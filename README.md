# Command-Conditioned Ant RL Workspace

This repository contains a Gymnasium + MuJoCo workflow for training and evaluating command-conditioned ant locomotion policies with SAC and PPO.

The project supports:
- Expert training per command (`stand`, `forward`, `left`, `right`)
- Curriculum training across command mixtures
- Single-model and multi-expert inference
- Flat and rough terrain evaluation
- Experiment automation and plotting scripts

## 1) Workspace Layout

- `gymnasium_ws/ant_rl/`: core package (environment wrappers, training loop, inference, artifacts/config handling)
- `gymnasium_ws/train_expert.py`: train one command-specific expert
- `gymnasium_ws/train_curriculum.py`: train one curriculum stage
- `gymnasium_ws/inference.py`: run one trained model interactively (keyboard command switching)
- `gymnasium_ws/multi_inference.py`: run multiple expert models with command-based switching
- `gymnasium_ws/rough_inference.py`: single-model inference on rough terrain
- `gymnasium_ws/rough_multi_inference.py`: multi-model inference on rough terrain
- `gymnasium_ws/experiment.py`: scripted experiment runner that logs metrics, trajectory, and recording
- `gymnasium_ws/evaluation/`: post-processing scripts and plots
- `gymnasium_ws/config.yaml`: training, algorithm, curriculum, and terrain defaults
- `gymnasium_ws/experiment-config.yaml`: scripted experiment setup and model paths

## 2) Environment Notes

This dev container is set up for EGL rendering:
- `MUJOCO_GL=egl` is expected
- Rendering is consumed via OpenCV/image buffers (`rgb_array`), which works well in container/WSL setups

If you run outside the dev container, ensure MuJoCo and OpenGL/EGL dependencies are installed.

## 3) Quick Start

From `/workspaces/gymnasium_ws`:

```bash
cd /workspaces/gymnasium_ws
python3 train_expert.py --command stand --algo sac --init-mode retrain
```

After a checkpoint exists, run inference:

```bash
python3 inference.py --model /workspaces/gymnasium_ws/artifacts/models/<your_model>.zip
```

Controls in inference windows:
- `W`: forward
- `A`: rotate left
- `D`: rotate right
- `ESC`: quit

## 4) Configuration

Main config file: `gymnasium_ws/config.yaml`

Important sections:
- `training`: global defaults (timesteps, number of vector envs, early-stop thresholds)
- `algorithms`: SAC/PPO defaults and device selection
- `curriculum`: named stages with command probability distributions
- `terrain.default_roughness`: default rough-terrain amplitude

Edit this file first when changing training behavior globally.

## 5) Training Workflows

### A) Expert training (single command)

```bash
python3 train_expert.py --command stand   --algo sac --init-mode resume
python3 train_expert.py --command forward --algo sac --init-mode resume
python3 train_expert.py --command left    --algo ppo --init-mode retrain
```

`--init-mode` options:
- `retrain`: start fresh
- `previous`: initialize from fallback/previous checkpoint (if configured)
- `resume`: continue current command checkpoint if available

### B) Curriculum training (mixed commands)

```bash
python3 train_curriculum.py --stage cur_stand   --algo sac --init-mode retrain
python3 train_curriculum.py --stage cur_forward --algo sac --init-mode previous
python3 train_curriculum.py --stage cur_turns   --algo ppo --init-mode resume
```

Stage names must match `config.yaml -> curriculum.stages[*].name`.

## 6) Inference Workflows

### A) Single-model inference (flat/normal env)

```bash
python3 inference.py --model /workspaces/gymnasium_ws/artifacts/models/<model>.zip
```

### B) Multi-expert inference

1. Edit `MODEL_PATHS` in `gymnasium_ws/multi_inference.py`
2. Run:

```bash
python3 multi_inference.py
```

### C) Rough-terrain inference

Single model:

```bash
python3 rough_inference.py --model /workspaces/gymnasium_ws/artifacts/models/<model>.zip --terrain-roughness 0.35
```

Multi model:

```bash
python3 rough_multi_inference.py --terrain-roughness 0.35 --terrain-seed 123
```

## 7) Experiment Runner (Benchmark-Style Evaluation)

Script: `gymnasium_ws/experiment.py`

This script executes a fixed command schedule from `experiment-config.yaml`, then writes:
- `metrics.csv` (reward over timestep)
- `trajectory.tum` (pose trajectory)
- `recording.mp4`

Example:

```bash
python3 experiment.py --config /workspaces/gymnasium_ws/experiment-config.yaml --experiment sac-multi-policy-flat
python3 experiment.py --config /workspaces/gymnasium_ws/experiment-config.yaml --experiment sac-single-policy-rough
```

Before running, ensure model paths in `experiment-config.yaml` exist.

## 8) Evaluation and Plotting

### A) Experiment reward plot

```bash
python3 evaluation/plot-experiment-metrics.py \
	evaluation/results/sac-single-policy-rough/metrics.csv \
	evaluation/results/sac-multi-policy-rough/metrics.csv
```

### B) Trajectory-derived plot (z/roll/pitch)

```bash
python3 evaluation/plot-trajectory-info.py \
	evaluation/results/sac-single-policy-rough/trajectory.tum \
	evaluation/results/sac-multi-policy-rough/trajectory.tum \
	--config /workspaces/gymnasium_ws/experiment-config.yaml
```

### C) Training curve comparison

```bash
python3 evaluation/training/plot-training-curves.py \
	--sac evaluation/training/expert_sac/training_forward_*.csv \
	--ppo evaluation/training/expert_ppo/training_forward_*.csv
```

### D) Aggregate training metrics

```bash
python3 evaluation/training/calc-training-metrics.py \
	--sac_dir evaluation/training/expert_sac \
	--ppo_dir evaluation/training/expert_ppo
```

## 9) Artifact Locations

Configured in `config.yaml`:
- Models: `gymnasium_ws/artifacts/models/`
- Curves: `gymnasium_ws/artifacts/curves/`
- Training CSVs: `gymnasium_ws/artifacts/training/`
- Run log: `gymnasium_ws/artifacts/log.csv`
- Experiment results: `gymnasium_ws/evaluation/results/`

## 10) Utility Scripts

- `benchmark_device.py`: compare CPU vs auto-device throughput for SAC/PPO
- `vis_cpu_gpu.py`: plot benchmark CSV output
- `hyperparam_gridsearch.py`: small search over selected SAC/PPO hyperparameters
- `vis_gridsearch_results.py`: plot grid-search outcomes
- `record_clips.py`: generate MP4/GIF clips for command experts


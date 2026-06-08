# Legged Lab — AMP Walking

Isaac Lab extension for legged robot reinforcement learning with AMP (Adversarial Motion Priors) on Unitree G1.

## Overview

- **Training:** Isaac Lab + RSL-RL AMP (`scripts/rsl_rl/train.py`)
- **Deploy / eval:** LeRobot pipeline (`amp-rollout`) for sim2sim and sim2real

## Environments

Use two separate conda environments:

| Environment | Python | Purpose |
|-------------|--------|---------|
| `isaaclab` (or your Isaac env) | 3.10 | Training and ONNX export |
| `lerobot-amp` | 3.12 | LeRobot deploy (sim2sim / sim2real) |

### Training environment (Isaac Lab)

Follow [legged_lab](https://github.com/zitongbai/legged_lab) setup instructions.

```bash
python scripts/rsl_rl/train.py --task Isaac-Locomotion-AMP-G1-v0 --headless
```

### Deploy environment (LeRobot)

```bash
conda create -y -n lerobot-amp python=3.12
conda activate lerobot-amp

# Unitree SDK + LeRobot (see https://huggingface.co/docs/lerobot/unitree_g1)
pip install lerobot[unitree_g1] onnxruntime pyyaml

# Install AMP plugins from this repo
pip install -e packages/lerobot_policy_amp_velocity
pip install -e packages/lerobot_robot_amp_g1
```

## Training → Deploy workflow

### 1. Export ONNX from checkpoint

```bash
python scripts/rsl_rl/play.py --task Isaac-Locomotion-AMP-G1-v0 --checkpoint <path/to/model.pt>
```

Add `--export-lerobot` to also produce a LeRobot `pretrained_model/` directory:

```bash
python scripts/rsl_rl/play.py --task Isaac-Locomotion-AMP-G1-v0 --checkpoint <path/to/model.pt> --export-lerobot
```

Or export manually:

```bash
python scripts/lerobot/export_amp_policy.py \
  --onnx deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx \
  --deploy-yaml deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy_1.yaml
```

### 2. Validate deploy config

```bash
python scripts/verify_deploy_yaml.py deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy_1.yaml
python scripts/lerobot/verify_obs_parity.py
```

### 3. Sim2sim

Uses LeRobot's built-in G1 MuJoCo sim over DDS (`lo`):

```bash
./scripts/lerobot/sim2sim.sh
```

Or directly:

```bash
amp-rollout \
  --strategy.type=base \
  --policy.path=deploy/robots/g1_29dof/config/policy/velocity/v0/exported/pretrained_model \
  --policy.type=amp_velocity \
  --robot.type=amp_g1 \
  --robot.is_simulation=true \
  --robot.network=lo \
  --robot.deploy_yaml=deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy_1.yaml \
  --fps=50 \
  --duration=300
```

### 4. Sim2real

```bash
./scripts/lerobot/sim2real.sh
# or: NETWORK=eth0 ./scripts/lerobot/sim2real.sh
```

## FSM keyboard controls

| Key | Action |
|-----|--------|
| ↑ | PASSIVE → FIXSTAND |
| → | FIXSTAND → VELOCITY (policy active) |
| ↓ | Any → PASSIVE |
| W/S | Forward / backward |
| A/D | Strafe left / right |
| Q/E | Turn left / right |
| Space | Zero velocity command |

Policy inference runs only in **VELOCITY** state (50 Hz). A background thread republishes motor commands at 500 Hz.

## AMP policy contract

| Property | Value |
|----------|-------|
| Observation | 585-dim (117/step × 5 history) |
| Action | 29-dim raw joint positions (policy order) |
| Post-process | `target = raw * 0.25 + offset`, remap via `joint_ids_map` |
| Control rate | 50 Hz policy, 500 Hz command stream |

## Package layout

```
packages/
  lerobot_policy_amp_velocity/   # LeRobot BYOP plugin (ONNX policy + obs builder)
  lerobot_robot_amp_g1/          # Custom G1 robot (DDS, FSM, safety)
scripts/lerobot/
  export_amp_policy.py           # Isaac Lab → LeRobot model export
  verify_obs_parity.py           # Obs builder parity check
  sim2sim.sh / sim2real.sh       # Deploy wrappers
deploy/robots/g1_29dof/config/policy/velocity/v0/
  exported/policy.onnx
  exported/pretrained_model/     # LeRobot model dir
  params/deploy_1.yaml
```

## References

- [LeRobot — Bring Your Own Policies](https://huggingface.co/docs/lerobot/bring_your_own_policies)
- [LeRobot — Unitree G1](https://huggingface.co/docs/lerobot/unitree_g1)
- [legged_lab](https://github.com/zitongbai/legged_lab)
- [RSL-RL AMP branch](https://github.com/zitongbai/rsl_rl/tree/feature/amp)

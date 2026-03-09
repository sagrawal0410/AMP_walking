#
# 🤖 Legged Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.1-silver)](https://isaac-sim.github.io/IsaacLab/v2.3.1/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## 📖 Overview

This repository is an extension for legged robot reinforcement learning based on Isaac Lab, which allows to develop in an isolated environment, outside of the core Isaac Lab repository. The RL algorithm is based on a [forked RSL-RL library](https://github.com/zitongbai/rsl_rl/tree/feature/amp). 

**Key Features:**

- `DeepMimic` for humanoid robots, including Unitree G1.
- `AMP` Adversarial Motion Priors (AMP) for humanoid robots, including Unitree G1. We suggest retargeting the human motion data by [GMR](https://github.com/YanjieZe/GMR).

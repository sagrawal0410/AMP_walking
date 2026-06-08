---
library_name: lerobot
tags:
- robotics
- amp_velocity
- unitree_g1
---

# AMP Velocity Locomotion Policy

Exported from Isaac Lab AMP training for Unitree G1 29-DOF.

- Observation: 585-dim proprioceptive vector (5-step history)
- Action: 29-dim joint position targets
- Control rate: 50 Hz

Deploy with:

```bash
amp-rollout \
  --strategy.type=base \
  --policy.path=/Users/shauryaagrawal/AMP_walking/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/pretrained_model \
  --policy.type=amp_velocity \
  --robot.type=amp_g1 \
  --robot.is_simulation=true \
  --fps=50
```

# AMP Policy Debug Instrumentation Guide

## Overview
This guide explains how to use the comprehensive debugging instrumentation added to diagnose AMP policy deployment issues.

## Quick Start

### 1. Enable Debug Logging
```bash
export DEPLOY_DEBUG=1
```

### 2. Enable Observation/Action Dumps
```bash
export DEPLOY_DUMP=1
export DEPLOY_DUMP_DIR=/tmp/amp_debug  # Optional, defaults to /tmp/amp_debug
```

### 3. Run Your Controller
```bash
cd deploy/robots/g1_29dof/build
./g1_ctrl --network lo
```

### 4. View Logs
Debug logs will appear every 50 policy steps (~1 second at 50Hz). Look for `[DEBUG]` tags.

## What Gets Logged

### Observation Terms (Every 50 Steps)
- **base_ang_vel**: Stats, first 3 values, NaN/Inf checks
- **root_local_rot_tan_norm**: Stats, orthonormality checks (||tan||, ||norm||, tan·norm)
- **keyboard_velocity_commands**: Stats, command values
- **joint_pos**: Stats, max|q - q_default|, joint mapping info
- **joint_vel**: Stats, max|dq| (should be near 0 when standing)
- **last_action**: Stats, saturation count (|a|>0.95)
- **key_body_pos_b**: Per-body xyz positions, FK validation

### History Stacking (Every 50 Steps)
- History length for each term
- Buffer size and per-step dimensions
- First 3 values of each timestep slice (oldest->newest)
- Confirms history order is correct

### Final 585-Dim Vector (Every 50 Steps)
- Total size verification (must be 585)
- Slice-by-slice stats for all 7 terms
- Global stats (min/max/mean/std/L2)
- NaN/Inf checks

### Control Rate (Every 50 Steps)
- Actual policy Hz vs expected Hz
- Warning if mismatch > 5Hz

### Joint Map (At Startup)
- Full joint_ids_map array
- Mapping checksum
- Duplicate/out-of-range validation

### ONNX Input/Output (Every 50 Steps)
- Input name, expected size, actual size
- Hard error if size mismatch (no padding)
- Output action stats and saturation count

## Dump Files

When `DEPLOY_DUMP=1`, files are written to `DEPLOY_DUMP_DIR`:
- `obs_obs_step_XXXXXX.csv`: Full 585-dim observation vector
- `act_step_XXXXXX.csv`: Raw action output from ONNX

Only first 200 steps are dumped to avoid disk bloat.

## Comparing with Python

Use the comparison script:
```bash
python scripts/compare_obs_dumps.py \
    --cpp-dump /tmp/amp_debug/obs_obs_step_000001.csv \
    --python-dump path/to/python_obs.npy
```

## Interpretation Guide

### If rot6 tan/norm norms not ~1.0
→ Quaternion/rotation conversion is wrong. Check:
- Quaternion order (wxyz vs xyzw)
- Yaw extraction formula
- Rotation matrix column selection

### If key_body_pos_b near zeros or huge
→ Wrong body mapping or FK not working. Check:
- Body names match training config
- FK implementation is correct
- Base frame transform (R^T*(p_body - p_base))

### If actions slice != previous raw action
→ Action observation semantics mismatch. Check:
- `last_action` uses `action_manager->action()` (post-processed)
- Should use raw network output before scale/offset

### If many actions saturated during standing
→ Normalization/scale/order mismatch. Check:
- Observation normalization matches training
- Action scaling/offset matches training
- Observation order matches training

### If policy Hz differs from training
→ Control rate mismatch causing twitch. Check:
- `step_dt` in deploy.yaml matches training `sim.dt * decimation`
- Policy thread timing is correct
- Consider setting `FORCE_POLICY_HZ=50` env var (if implemented)

## File Locations

### C++ Files Modified
- `deploy/include/isaaclab/utils/debug_utils.h`: Debug utilities
- `deploy/include/isaaclab/envs/mdp/observations/observations.h`: Term instrumentation
- `deploy/include/isaaclab/manager/observation_manager.h`: History/assembly instrumentation
- `deploy/include/isaaclab/envs/manager_based_rl_env.h`: Timing/joint map instrumentation
- `deploy/include/isaaclab/algorithms/algorithms.h`: ONNX I/O instrumentation
- `deploy/include/isaaclab/manager/manager_term_cfg.h`: History buffer initialization
- `deploy/robots/g1_29dof/src/State_RLBase.cpp`: Keyboard command instrumentation

### Python Scripts
- `scripts/compare_obs_dumps.py`: Compare C++ vs Python dumps

## Call Chain

```
State_RLBase::enter()
  └─> Policy thread: env->step() [every step_dt]
      └─> ManagerBasedRLEnv::step()
          ├─> robot->update() [update state]
          ├─> observation_manager->compute()
          │   └─> compute_group()
          │       ├─> For each term: term.add(term.func(...)) [add to history]
          │       └─> Concatenate history: term.get() [oldest->newest]
          ├─> alg->act(obs) [ONNX inference]
          │   └─> OrtRunner::act()
          │       ├─> Validate input sizes
          │       ├─> session->Run()
          │       └─> Validate output
          └─> action_manager->process_action(action)
```

## Next Steps

1. Run with `DEPLOY_DEBUG=1` and collect logs
2. Check for warnings/errors in logs
3. Compare slice stats with expected values
4. Use dump files to compare with Python if available
5. Fix issues based on interpretation guide

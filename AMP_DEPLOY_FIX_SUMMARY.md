# AMP Policy Deploy Fix Summary

## Overview
This document summarizes the fixes implemented to integrate LeggedLab AMP policies into Unitree's deploy stack (g1_ctrl + deploy.yaml).

## Part A: LeggedLab Export Pipeline Fixes ✅

### A1-A2: Export Function Updates
**File**: `source/legged_lab/legged_lab/utils/export_deploy_cfg.py`

**Changes**:
1. Added `obs_order` export to preserve exact observation term ordering
2. Added AMP-specific observations to `registered_observations`:
   - `root_local_rot_tan_norm`
   - `key_body_pos_b`
   - `joint_vel` (separate from `joint_vel_rel`)
3. Fixed observation name mapping to handle AMP vs velocity tasks:
   - AMP uses `joint_pos`/`joint_vel` (not `joint_pos_rel`/`joint_vel_rel`)
   - AMP uses `keyboard_velocity_commands` (not `velocity_commands`)
4. Added special handling for `key_body_pos_b` params to ensure `body_names` are exported correctly

**File**: `scripts/rsl_rl/train.py`

**Changes**:
- Added `env.reset()` call before `export_deploy_cfg()` to ensure all observations are initialized
- Critical for AMP policies with `key_body_pos_b` and `root_local_rot_tan_norm`

### A3-A4: Validation Scripts ✅

**Created**:
1. `scripts/verify_deploy_yaml.py` - Verifies deploy.yaml correctness:
   - Checks `obs_order` matches required list exactly
   - Validates all observations have `history_length == 5`
   - Verifies total observation dimension == 585
   - Checks `key_body_pos_b` has correct `body_names`

2. `scripts/check_onnx_io.py` - Checks ONNX model I/O:
   - Verifies input dimension == 585
   - Prints input/output shapes

3. `scripts/dump_amp_obs.py` - Dumps Python observations:
   - Creates ground truth observation dumps for comparison
   - Saves individual term observations and full flattened vector

## Part B: Unitree Deploy C++ Code Fixes ✅

### B1-B2: AMP Observation Registration ✅

**File**: `deploy/include/isaaclab/envs/mdp/observations/observations.h`

**Added Observations**:

1. **`joint_vel`** - Joint velocities (absolute, not relative)
   - Similar to `joint_pos`, supports `joint_ids` filtering
   - Used by AMP policies (velocity uses `joint_vel_rel`)

2. **`root_local_rot_tan_norm`** - Root rotation in local frame (yaw-removed) as tan/norm representation
   - Implementation matches Python exactly:
     - Extracts yaw quaternion from root orientation
     - Removes yaw: `root_quat_local = yaw_quat^{-1} * root_quat_w`
     - Converts to rotation matrix
     - Extracts first column (tan) and third column (norm)
     - Returns 6D: `[tan.x, tan.y, tan.z, norm.x, norm.y, norm.z]`

3. **`key_body_pos_b`** - Key body positions in base frame
   - **⚠️ PLACEHOLDER IMPLEMENTATION**: Currently returns zeros
   - **TODO for Real Robot**: Implement Forward Kinematics (FK)
   - Expected body names (from `g1_amp_env_cfg.py`):
     - `left_ankle_roll_link`
     - `right_ankle_roll_link`
     - `left_wrist_yaw_link`
     - `right_wrist_yaw_link`
     - `left_shoulder_roll_link`
     - `right_shoulder_roll_link`
   - Returns 18D: `[LA.xyz, RA.xyz, LW.xyz, RW.xyz, LS.xyz, RS.xyz]`

### B3-B4: Observation Manager Order Preservation ✅

**File**: `deploy/include/isaaclab/manager/observation_manager.h`

**Changes**:
1. Added `obs_order` parsing from deploy.yaml
2. Modified `_prepare_group_terms()` to:
   - Parse `obs_order` from config
   - Build observations in exact order specified by `obs_order`
   - Fall back to map order if `obs_order` not specified (backward compatible)
3. Added necessary includes: `<map>`, `<algorithm>`, `<spdlog/spdlog.h>`

## Part C: Validation & Testing

### C1: Python Observation Dump ✅
- Script created: `scripts/dump_amp_obs.py`
- Dumps ground truth observations from Python environment

### C2: C++ Observation Dump ⚠️ PENDING
- Not yet implemented
- Can be added later for debugging

## Expected Observation Structure

### Ground Truth (585 dims total, history_length=5):

1. `base_ang_vel`: 15 dims (3 * 5)
2. `root_local_rot_tan_norm`: 30 dims (6 * 5)
3. `keyboard_velocity_commands`: 15 dims (3 * 5)
4. `joint_pos`: 145 dims (29 * 5)
5. `joint_vel`: 145 dims (29 * 5)
6. `last_action`: 145 dims (29 * 5)
7. `key_body_pos_b`: 90 dims (18 * 5)

**Total**: 585 dims

## Critical TODOs for Real Robot Deployment

### 1. Forward Kinematics Implementation for `key_body_pos_b`
**Status**: ⚠️ **CRITICAL - NOT IMPLEMENTED**

The `key_body_pos_b` observation currently returns zeros. For real robot deployment, this **MUST** be implemented using Forward Kinematics:

```cpp
// Pseudo-code for real robot FK implementation:
// 1. Load URDF model
// 2. For each key body name:
//    - Compute FK transform from base to key body using joint angles
//    - Extract position: p_body_base = FK(base->key_link).translation
// 3. Concatenate in KEY_BODY_NAMES order
```

**Implementation Location**: `deploy/include/isaaclab/envs/mdp/observations/observations.h` - `key_body_pos_b` function

**Build Flag**: Consider adding `SIM_MODE` vs `REAL_MODE` flag:
- `SIM_MODE`: Use MuJoCo `xmat`/`xpos` (for sim2sim)
- `REAL_MODE`: Use FK (for real robot)
- Fail with error if `REAL_MODE` has no FK

### 2. MuJoCo Body Name Mapping (for sim2sim)
**Status**: ⚠️ **NEEDS VERIFICATION**

If using MuJoCo for sim2sim, verify body names match:
- MuJoCo body names might differ from Isaac Lab link names
- May need mapping table: `"left_ankle_roll_link" -> "left_ankle_roll"`
- Add logging to print missing names and available bodies

### 3. Action Term Definition Verification
**Status**: ⚠️ **NEEDS VERIFICATION**

Verify that `last_action` observation matches training:
- Check if training records normalized actions `[-1, 1]` or applied joint targets
- Ensure C++ implementation matches Python exactly

## Testing Checklist

- [ ] Run `verify_deploy_yaml.py` on exported deploy.yaml
- [ ] Run `check_onnx_io.py` on exported policy.onnx
- [ ] Run `dump_amp_obs.py` to create ground truth
- [ ] Compare C++ observations with Python dump (when FK implemented)
- [ ] Test ONNX inference match between Python and C++
- [ ] Test sim2sim with keyboard commands = [0,0,0] (stable stance)
- [ ] Test with small commands (0.05 m/s) only

## Files Modified

### Python:
- `source/legged_lab/legged_lab/utils/export_deploy_cfg.py`
- `scripts/rsl_rl/train.py`

### C++:
- `deploy/include/isaaclab/envs/mdp/observations/observations.h`
- `deploy/include/isaaclab/manager/observation_manager.h`

### New Scripts:
- `scripts/verify_deploy_yaml.py`
- `scripts/check_onnx_io.py`
- `scripts/dump_amp_obs.py`

## Notes

1. **Quaternion Format**: Isaac Lab uses `[w, x, y, z]` format (confirmed in code comments)
2. **Observation Order**: Critical for AMP policies - must match exactly
3. **History Length**: All terms use `history_length=5` (flattened: oldest->newest)
4. **Padding**: No padding needed - observations should match exactly 585 dims

## Next Steps

1. **Implement FK for `key_body_pos_b`** (critical for real robot)
2. **Test with actual AMP policy** to verify end-to-end
3. **Add C++ observation dump** for debugging
4. **Verify action term** matches training exactly

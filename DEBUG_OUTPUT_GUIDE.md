# Debug Output Guide for AMP Policy Deployment

## Overview
This guide explains all debug outputs and how to diagnose issues with the AMP policy on the real robot.

---

## 1. STARTUP DEBUG (First 10 steps)

### What it shows:
Each observation term's values at startup, helping verify the observation pipeline is correct.

### Format:
```
[STARTUP DEBUG] Step 1/10 ========================================
[STARTUP DEBUG] Term[0] 'base_ang_vel': size=3
[STARTUP DEBUG]   First 3 values: [0.0319, -0.1302, 0.0026]
[STARTUP DEBUG]   Stats: min=-0.1302, max=0.0319, mean=-0.0319
```

### What to look for:

**✅ GOOD:**
- `base_ang_vel`: Small values (< 1.0 rad/s) when robot is standing still
- `root_local_rot_tan_norm`: First value ≈ 0.99-1.0 (upright robot), last value ≈ -0.1 to -0.2 (slight pitch)
- `keyboard_velocity_commands`: Should match your key presses (e.g., [0.15, 0, 0] for 'w')
- `joint_pos`: Reasonable joint angles (typically -0.5 to 1.5 rad)
- `joint_vel`: Small velocities (< 2.0 rad/s) when standing
- `key_body_pos_b`: Ankles at z ≈ -0.75m (below pelvis), wrists/shoulders at reasonable positions

**❌ BAD:**
- `base_ang_vel`: Large values (> 5 rad/s) = robot spinning/falling
- `root_local_rot_tan_norm`: First value < 0.9 = robot tilted significantly
- `keyboard_velocity_commands`: Always [0,0,0] = commands not updating
- `joint_pos`: Extreme values (> 3 rad) = joint limits exceeded
- `key_body_pos_b`: All zeros or huge values (> 10m) = FK calculation broken

---

## 2. OBS SIZE DEBUG & OBS DEBUG

### What it shows:
The final observation vector sent to the policy (should be 585 dimensions).

### Format:
```
[OBS SIZE DEBUG] Step 1: obs size = 585 (expected 585)
[OBS DEBUG] First 20 values: [0.0319, -0.1302, 0.0026, 0.0319, -0.1302, 0.0026, ...]
```

### What to look for:

**✅ GOOD:**
- Size = 585 exactly
- First 15 values: Should be `base_ang_vel` repeated 5 times (history_length=5)
- Values 16-21: Should be `root_local_rot_tan_norm` (first 6 values)
- Values change smoothly over time

**❌ BAD:**
- Size ≠ 585 = observation mismatch (policy will fail!)
- All zeros = observations not being computed
- NaN or Inf = numerical error
- Values jumping wildly = observation order mismatch

---

## 3. ACTION DEBUG (Every 50 steps)

### What it shows:
Raw policy outputs and processed actions after scaling/clipping.

### Format:
```
[ACTION DEBUG] Raw: max=1.4492, saturated=3/29, Processed: max=1.0676 rad
[ACTION DEBUG] Raw[0:5]: [-0.298, -0.225, 0.006, 0.526, -0.285, -0.184]
[ACTION DEBUG] Proc[0:5]: [-0.175, -0.156, 0.002, 0.131, -0.071, -0.046]
```

### Key Metrics:

#### **Raw Actions:**
- **Range**: Policy outputs are typically in [-1, 1] or [-2, 2] range
- **Max value**: Should be < 2.0 for normal operation
- **Saturated count**: Number of actions hitting ±1.0 limit

#### **Processed Actions:**
- **Max value**: After scaling by 0.25, max should be < 0.5 rad typically
- **After offset**: Final joint positions sent to robot

### What to look for:

**✅ GOOD (Standing Still):**
- Raw max: 0.5 - 1.5
- Saturated: 0-5/29
- Processed max: 0.3 - 0.8 rad
- Actions oscillate slightly but don't grow unbounded

**✅ GOOD (Walking):**
- Raw max: 1.0 - 2.5
- Saturated: 3-8/29
- Processed max: 0.5 - 1.2 rad
- Actions change smoothly with gait

**❌ BAD (Unstable):**
- Raw max: > 3.0 = Policy outputting extreme values
- Saturated: > 15/29 = Policy struggling, many actions at limits
- Processed max: > 1.5 rad = Very large joint movements
- Actions growing unbounded = Feedback instability

**❌ BAD (Not Moving):**
- Raw max: < 0.3 = Policy outputting very small actions
- All actions near zero = Policy thinks robot is already at target
- Actions don't change with commands = Policy not responding to velocity commands

---

## 4. ROT DEBUG (Every 100 steps)

### What it shows:
IMU quaternion processing for orientation observation.

### Format:
```
[ROT DEBUG] Raw IMU quaternion (wxyz): w=0.9986, x=-0.0489, y=-0.0190, z=0.0094
[ROT DEBUG] After yaw removal: tan=[0.9993,0.0000,0.0371], norm=[-0.0369,0.0980,0.9945]
```

### What to look for:

**✅ GOOD (Upright Robot):**
- `w` ≈ 0.99-1.0 (quaternion is normalized, w is largest component)
- `tan[0]` ≈ 1.0 (forward direction in base frame)
- `tan[2]` ≈ 0.0-0.1 (slight pitch)
- `norm[2]` ≈ 0.99-1.0 (upward direction)

**❌ BAD:**
- `w` ≈ 0 = Quaternion order might be wrong (should be wxyz, not xyzw)
- `tan` or `norm` have large values (> 0.5) = Robot tilted significantly
- Values jumping = IMU noise or processing error

---

## 5. FK DEBUG (Every 100 steps)

### What it shows:
Forward kinematics calculations for key body positions.

### Format:
```
[FK DEBUG] Joint positions in POLICY order:
[FK DEBUG]   [0] L_hip_pitch=0.1421, [1] R_hip_pitch=-0.0067, [2] waist_yaw=-0.0764
[FK DEBUG] left_ankle_roll_link: [-0.0794, 0.1625, -0.7503]
```

### What to look for:

**✅ GOOD:**
- Joint positions: Reasonable angles (-1.5 to 1.5 rad typically)
- Ankle z-position: ≈ -0.75m (ankles ~75cm below pelvis)
- Ankle x/y: Small values (< 0.2m) when standing
- Wrist positions: Reasonable (arms not contorted)
- Shoulder positions: Reasonable height (z ≈ 0.28-0.30m)

**❌ BAD:**
- Ankle z > -0.5m = Robot crouching too much or FK error
- Ankle z < -1.0m = Robot stretched or FK error
- All positions zero = FK not working
- Positions > 10m = FK coordinate frame error

---

## 6. Command Updated Messages

### Format:
```
[info] Key detected: 'w' -> Command will be generated
[info] Command updated: [0.300, 0.000, 0.000]
```

### What to look for:

**✅ GOOD:**
- Commands update when keys pressed
- Values match your `key_commands` map
- Commands persist (don't reset to zero) when key released

**❌ BAD:**
- Commands always [0,0,0] = Keyboard input not working
- Commands reset to zero immediately = Command persistence broken
- Commands don't match key presses = Key mapping wrong

---

## How to Diagnose Action Issues

### Step 1: Check if Policy is Outputting Actions

Look at `[ACTION DEBUG] Raw`:
- If **all zeros** → Policy not running or observation mismatch
- If **very small** (< 0.1) → Policy thinks robot is already at target
- If **reasonable** (0.5-2.0) → Policy is working, check processing

### Step 2: Check if Actions are Being Processed Correctly

Compare `Raw[0:5]` vs `Proc[0:5]`:
- Raw should be in [-1, 1] or [-2, 2] range
- Processed = Raw × scale (0.25) + offset
- If processed values are wrong → Check `deploy.yaml` scale/offset

### Step 3: Check if Actions are Too Small/Large

**Too Small (Robot Not Moving):**
- Raw max < 0.3 → Policy outputting tiny actions
- **Possible causes:**
  - Command too small (try increasing to 0.3-0.5 m/s)
  - Policy trained with different action scaling
  - Observation normalization mismatch

**Too Large (Robot Unstable):**
- Raw max > 3.0 → Policy outputting extreme actions
- Saturated > 15/29 → Policy struggling
- **Possible causes:**
  - Sim2real gap (dynamics mismatch)
  - Observation noise/distortion
  - PD gains too high/low
  - Action smoothing too low

### Step 4: Check Action Smoothing

Action smoothing is applied in `State_RLBase.cpp`:
```cpp
smoothed_action[i] = ACTION_SMOOTHING * smoothed_action[i] + (1.0f - ACTION_SMOOTHING) * action_val;
```

- **ACTION_SMOOTHING = 0.0**: No smoothing (most responsive, can be jittery)
- **ACTION_SMOOTHING = 0.5**: Moderate smoothing (good for sim2real)
- **ACTION_SMOOTHING = 0.8**: Heavy smoothing (very stable but slow response)

**If robot oscillates:** Increase smoothing (0.5 → 0.7)
**If robot doesn't respond:** Decrease smoothing (0.5 → 0.3)

### Step 5: Check Command Magnitude

Commands are in `State_RLBase.cpp`:
```cpp
{"w", {0.15f, 0.0f, 0.0f}}  // Forward velocity in m/s
```

**If robot doesn't move:**
- Try increasing to 0.3-0.5 m/s
- Check if `keyboard_velocity_commands` observation shows the command
- Verify policy was trained with similar command ranges

**If robot falls over:**
- Reduce to 0.1-0.2 m/s
- Check if actions are saturating (> 15/29)
- May need to retrain policy or tune PD gains

---

## Diagnostic Checklist

### Robot Not Moving Forward:

- [ ] Check `[ACTION DEBUG] Raw max` - is it > 0.3?
- [ ] Check `keyboard_velocity_commands` - does it show [0.15, 0, 0] when pressing 'w'?
- [ ] Check `[ACTION DEBUG] Proc[0:5]` - are processed actions reasonable (0.1-0.5 rad)?
- [ ] Check action smoothing - try reducing from 0.5 to 0.3
- [ ] Check command magnitude - try increasing from 0.15 to 0.3 m/s
- [ ] Check if actions are being applied - verify `lowcmd` is being sent to robot

### Robot Unstable/Oscillating:

- [ ] Check `[ACTION DEBUG] Raw max` - is it > 3.0?
- [ ] Check `saturated` count - is it > 15/29?
- [ ] Check `base_ang_vel` - is it oscillating?
- [ ] Check `root_local_rot_tan_norm` - is robot tilting?
- [ ] Increase action smoothing (0.5 → 0.7)
- [ ] Reduce command magnitude (0.3 → 0.15 m/s)
- [ ] Check PD gains in `deploy.yaml` - may need to reduce stiffness

### Robot Falls Over Immediately:

- [ ] Check `[OBS SIZE DEBUG]` - is obs size = 585?
- [ ] Check `[OBS DEBUG]` first 20 values - do they look reasonable?
- [ ] Check `[ROT DEBUG]` - is quaternion order correct (w ≈ 0.99)?
- [ ] Check `[FK DEBUG]` - are FK positions reasonable?
- [ ] Check `[ACTION DEBUG] Raw` - are actions reasonable or extreme?
- [ ] Verify `obs_order` in `deploy.yaml` matches training

---

## Expected Values for Stable Operation

### Standing Still (zero commands):
- Raw actions max: 0.5 - 1.5
- Saturated: 0-5/29
- Processed max: 0.3 - 0.8 rad
- Base ang vel: < 0.5 rad/s
- Root rot tan[0]: 0.98 - 1.0

### Walking Forward (0.3 m/s):
- Raw actions max: 1.0 - 2.5
- Saturated: 3-10/29
- Processed max: 0.5 - 1.2 rad
- Base ang vel: < 1.0 rad/s
- Root rot tan[0]: 0.95 - 1.0

### Walking Backward (-0.3 m/s):
- Similar to forward but may be slightly less stable
- If much worse, may need to reduce command or retrain

---

## Quick Fixes

1. **Robot not moving**: Increase command from 0.15 → 0.3 m/s
2. **Robot oscillating**: Increase smoothing from 0.5 → 0.7
3. **Robot falling**: Reduce command from 0.3 → 0.15 m/s, increase smoothing to 0.7
4. **Actions too small**: Check if observations are correct, verify command is being sent
5. **Actions too large**: Increase smoothing, reduce command, check PD gains

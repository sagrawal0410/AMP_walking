# Forward Kinematics Diagnostic Guide

This guide helps you diagnose if the FK implementation is working correctly.

## Quick Test Methods

### Method 1: C++ Standalone Test (Fastest)

1. **Build the test executable:**
```bash
cd deploy/robots/g1_29dof/build
make test_fk
```

2. **Run the test:**
```bash
./test_fk
```

This will:
- Test FK with zero pose (all joints = 0)
- Test FK with standing pose
- Test FK with extreme angles
- Check symmetry (left vs right should be symmetric)
- Check position ranges (bodies should be within reasonable bounds)
- Report any NaN/Inf values

### Method 2: Python Simulation Comparison

1. **Run Python test:**
```bash
python scripts/test_fk_python.py --task Isaac-G1-Amp-v0 --num_tests 5
```

This will:
- Create a simulation environment
- Get joint positions from simulation
- Compute key_body_pos_b observation
- Print positions for comparison with C++

### Method 3: Runtime Test (During Robot Operation)

Add debug logging to the FK function in `observations.h`:

```cpp
REGISTER_OBSERVATION(key_body_pos_b)
{
    // ... existing code ...
    
    // Add debug output
    static int call_count = 0;
    if (call_count++ % 100 == 0) {  // Print every 100 calls
        for (size_t i = 0; i < num_key_bodies; ++i) {
            Eigen::Vector3f pos = computeKeyBodyPosition_G1(body_names[i], joint_pos);
            spdlog::info("FK[{}] {}: [{:.4f}, {:.4f}, {:.4f}]", 
                        call_count, body_names[i], pos.x(), pos.y(), pos.z());
        }
    }
    
    // ... rest of code ...
}
```

Then run your robot controller and check logs.

## What to Look For

### ✅ Good Signs:
- Positions are within reasonable ranges (ankles ~[-0.5, 0.5] in x/y, ~[-1.0, -0.5] in z)
- No NaN or Inf values
- Left/right symmetry in zero pose
- Positions change smoothly as joints move
- Positions match Python simulation (within ~0.01m)

### ❌ Warning Signs:
- NaN or Inf values → Check for division by zero or invalid transforms
- Positions are all zeros → FK not being called or wrong function
- Positions don't change with joint angles → Wrong joint mapping
- Asymmetry in zero pose → Wrong transform for left/right side
- Positions way outside expected range → Wrong kinematic chain or units

## Common Issues and Fixes

### Issue: All positions are zero
**Cause:** FK function not being called or joint_pos is empty
**Fix:** Check that `joint_pos` has 29 elements and contains valid angles

### Issue: Positions are NaN
**Cause:** Invalid quaternion or transform computation
**Fix:** Check quaternion normalization and transform composition

### Issue: Wrong positions (but not NaN)
**Cause:** Wrong joint order, wrong transforms, or wrong axes
**Fix:** 
- Verify joint order matches SDK order
- Check transform values match XML
- Verify joint axes (Y=pitch, X=roll, Z=yaw)

### Issue: Asymmetry
**Cause:** Wrong sign or transform for left vs right side
**Fix:** Check that left/right transforms have correct signs

## Expected Values (Zero Pose)

In zero pose (all joints = 0), approximate positions:

- **left_ankle_roll_link**:  [~0.0, ~0.06, ~-0.7]
- **right_ankle_roll_link**: [~0.0, ~-0.06, ~-0.7]
- **left_wrist_yaw_link**:   [~0.3, ~0.1, ~0.2]
- **right_wrist_yaw_link**:  [~0.3, ~-0.1, ~0.2]
- **left_shoulder_roll_link**: [~0.0, ~0.1, ~0.24]
- **right_shoulder_roll_link**: [~0.0, ~-0.1, ~0.24]

(Note: These are approximate - exact values depend on your robot's kinematic parameters)

## Comparing with Python

If you have access to Python simulation, you can compare:

1. Run Python test to get reference positions
2. Run C++ test with same joint angles
3. Compare outputs - they should match within ~0.01m

## Debugging Tips

1. **Test one body at a time:** Comment out other bodies to isolate issues
2. **Test with known poses:** Use zero pose or poses where you know expected positions
3. **Print intermediate transforms:** Add logging in `computeKeyBodyPosition_G1` to see transform chain
4. **Check joint order:** Verify joint_pos indices match SDK order
5. **Validate transforms:** Check that static transforms match XML file

## Next Steps

If FK is working:
- ✅ You should see reasonable positions
- ✅ Positions change smoothly with joint angles
- ✅ No NaN/Inf values

If FK has issues:
- Check the specific error messages from the test
- Compare with Python simulation if available
- Review the transform chain in `computeKeyBodyPosition_G1`
- Verify joint order and transform values match XML

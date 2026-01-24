#!/usr/bin/env python3
"""
Check ONNX model input/output dimensions.

Verifies that the policy expects 585-dim observation input.
"""

import argparse
import sys
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime not installed. Install with: pip install onnxruntime")
    sys.exit(1)

EXPECTED_INPUT_DIM = 585


def check_onnx_io(onnx_path: str, verbose: bool = True):
    """Check ONNX model input/output dimensions."""
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        print(f"[ERROR] ONNX file not found: {onnx_file}")
        return False
    
    try:
        session = ort.InferenceSession(str(onnx_file))
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}")
        return False
    
    # Get input info
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    if verbose:
        print("=" * 80)
        print("ONNX Model I/O Information")
        print("=" * 80)
    
    errors = []
    
    # Check inputs
    if len(inputs) == 0:
        errors.append("Model has no inputs")
    else:
        for i, inp in enumerate(inputs):
            shape = inp.shape
            if verbose:
                print(f"\nInput[{i}]: {inp.name}")
                print(f"  Shape: {shape}")
                print(f"  Type:  {inp.type}")
            
            # Calculate total dimension
            if len(shape) >= 2:
                # Usually [batch, features] or [batch, seq_len, features]
                if len(shape) == 2:
                    input_dim = shape[1]
                elif len(shape) == 3:
                    # Flattened: batch * seq_len * features
                    input_dim = shape[1] * shape[2]
                else:
                    # Flatten all dimensions except batch
                    import numpy as np
                    input_dim = int(np.prod(shape[1:]))
            else:
                input_dim = int(shape[0]) if len(shape) == 1 else 1
            
            if verbose:
                print(f"  Total input dim (excluding batch): {input_dim}")
            
            if input_dim != EXPECTED_INPUT_DIM:
                errors.append(
                    f"Input dimension mismatch!\n"
                    f"  Expected: {EXPECTED_INPUT_DIM}\n"
                    f"  Got:      {input_dim}"
                )
    
    # Check outputs
    if verbose:
        print("\n" + "-" * 80)
    
    for i, out in enumerate(outputs):
        shape = out.shape
        if verbose:
            print(f"\nOutput[{i}]: {out.name}")
            print(f"  Shape: {shape}")
            print(f"  Type:  {out.type}")
        
        if len(shape) >= 2:
            output_dim = shape[1]
        else:
            output_dim = int(shape[0]) if len(shape) == 1 else 1
        
        if verbose:
            print(f"  Total output dim (excluding batch): {output_dim}")
    
    if verbose:
        print("\n" + "=" * 80)
    
    # Report results
    if errors:
        print("\n[ERRORS]")
        for e in errors:
            print(f"  - {e}")
        return False
    
    if verbose:
        print("\n[SUCCESS] ONNX model I/O check passed!")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Check ONNX model I/O dimensions")
    parser.add_argument(
        "onnx_file",
        type=str,
        help="Path to policy.onnx file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed information",
    )
    args = parser.parse_args()
    
    success = check_onnx_io(args.onnx_file, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

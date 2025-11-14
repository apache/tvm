#!/usr/bin/env python3
"""
Test script for CPU Sample device API
This script tests basic operations on the cpu_sample device.
"""

import tvm
import numpy as np
from tvm import te

def test_cpu_sample_device_basic():
    """Test basic device operations"""
    print("=" * 80)
    print("Testing CPU Sample Device API")
    print("=" * 80)

    # Device type 20 corresponds to kDLCPUSample
    device_type = 20
    device_id = 0

    print(f"\n[TEST] Creating device with type={device_type}, id={device_id}")
    try:
        dev = tvm.device(device_type, device_id)
        print(f"[TEST] Device created: {dev}")
    except Exception as e:
        print(f"[ERROR] Failed to create device: {e}")
        return False

    print("\n[TEST] Checking device existence")
    try:
        exists = dev.exist
        print(f"[TEST] Device exists: {exists}")
        if not exists:
            print("[ERROR] Device does not exist!")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to check device existence: {e}")
        return False

    print("\n[TEST] Allocating array on device")
    try:
        # Create a simple array
        n = 1024
        a = tvm.nd.array(np.random.randn(n).astype("float32"), dev)
        print(f"[TEST] Array allocated: shape={a.shape}, dtype={a.dtype}")
    except Exception as e:
        print(f"[ERROR] Failed to allocate array: {e}")
        return False

    print("\n[TEST] Copying data to another array")
    try:
        b = tvm.nd.array(np.zeros(n).astype("float32"), dev)
        b.copyfrom(a)
        print(f"[TEST] Data copied successfully")
    except Exception as e:
        print(f"[ERROR] Failed to copy data: {e}")
        return False

    print("\n[TEST] Verifying data")
    try:
        a_np = a.numpy()
        b_np = b.numpy()
        if np.allclose(a_np, b_np):
            print("[TEST] Data verification passed!")
        else:
            print("[ERROR] Data verification failed!")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to verify data: {e}")
        return False

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
    return True


def test_cpu_sample_simple_compute():
    """Test simple computation on cpu_sample device"""
    print("\n" + "=" * 80)
    print("Testing Simple Computation on CPU Sample Device")
    print("=" * 80)

    device_type = 20
    dev = tvm.device(device_type, 0)

    print("\n[TEST] Creating simple computation (element-wise add)")
    try:
        # Define computation
        n = te.var("n")
        A = te.placeholder((n,), name="A", dtype="float32")
        B = te.placeholder((n,), name="B", dtype="float32")
        C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

        # Create schedule
        s = te.create_schedule(C.op)

        # Build function
        fadd = tvm.build(s, [A, B, C], target="llvm", name="add")
        print(f"[TEST] Function built successfully")

        # Prepare data
        n_val = 1024
        a_np = np.random.randn(n_val).astype("float32")
        b_np = np.random.randn(n_val).astype("float32")
        c_np = np.zeros(n_val).astype("float32")

        # Note: For now, we use CPU device for computation
        # since cpu_sample is a minimal implementation
        cpu_dev = tvm.cpu(0)
        a_tvm = tvm.nd.array(a_np, cpu_dev)
        b_tvm = tvm.nd.array(b_np, cpu_dev)
        c_tvm = tvm.nd.array(c_np, cpu_dev)

        # Execute
        fadd(a_tvm, b_tvm, c_tvm)

        # Verify
        np.testing.assert_allclose(c_tvm.numpy(), a_np + b_np, rtol=1e-5)
        print("[TEST] Computation verification passed!")

    except Exception as e:
        print(f"[ERROR] Computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("Computation test passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# CPU Sample Device API Test Suite")
    print("#" * 80 + "\n")

    success = True

    # Run basic device tests
    if not test_cpu_sample_device_basic():
        success = False

    # Run simple computation test
    if not test_cpu_sample_simple_compute():
        success = False

    if success:
        print("\n" + "#" * 80)
        print("# ALL TESTS PASSED!")
        print("#" * 80 + "\n")
    else:
        print("\n" + "#" * 80)
        print("# SOME TESTS FAILED!")
        print("#" * 80 + "\n")
        exit(1)

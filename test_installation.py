#!/usr/bin/env python3
"""
Simple test script to verify TVM installation.
Run this after installing TVM to ensure everything works correctly.
"""

try:
    import tvm
    print(f"✅ TVM imported successfully!")
    print(f"   Version: {tvm.__version__}")
    print(f"   Runtime only: {getattr(tvm, '_RUNTIME_ONLY', False)}")
    
    # Test basic functionality
    print("\n🔧 Testing basic functionality...")
    
    # Test device creation
    cpu_dev = tvm.cpu(0)
    print(f"   CPU device: {cpu_dev}")
    
    # Test NDArray creation
    import numpy as np
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    tvm_array = tvm.nd.array(data, device=cpu_dev)
    print(f"   NDArray created: {tvm_array}")
    print(f"   Shape: {tvm_array.shape}")
    print(f"   Dtype: {tvm_array.dtype}")
    
    # Test basic operations
    result = tvm_array + 1
    print(f"   Array + 1: {result.numpy()}")
    
    print("\n🎉 All basic tests passed! TVM is working correctly.")
    
except ImportError as e:
    print(f"❌ Failed to import TVM: {e}")
    print("   Make sure you have installed TVM correctly.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error during testing: {e}")
    print("   TVM imported but encountered an error during testing.")
    sys.exit(1)

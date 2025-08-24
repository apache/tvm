#!/usr/bin/env python3
"""Test basic Relax syntax."""

import tvm
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


@I.ir_module
class BasicModule:
    """Basic Relax module for testing syntax."""
    
    @T.prim_func
    def add(
        var_A: T.handle,
        var_B: T.handle,
        var_C: T.handle,
    ):
        n = T.int32()
        A = T.match_buffer(var_A, (n,), "float32")
        B = T.match_buffer(var_B, (n,), "float32")
        C = T.match_buffer(var_C, (n,), "float32")
        for i in T.grid(n):
            with T.block("add"):
                vi = T.axis.remap("S", [i])
                C[vi] = A[vi] + B[vi]

    @R.function
    def simple(x: R.Tensor(("n",), "float32")) -> R.Tensor(("n",), "float32"):
        return x

    @R.function
    def double(x: R.Tensor(("n",), "float32")) -> R.Tensor(("n",), "float32"):
        return x + x


def test_basic_syntax():
    """Test basic Relax syntax."""
    print("🧪 Testing basic Relax syntax...")
    
    try:
        # Get the IRModule
        ir_mod = BasicModule
        print(f"✓ IRModule created: {type(ir_mod)}")
        
        # Check functions
        functions = list(ir_mod.functions.keys())
        print(f"✓ Functions found: {functions}")
        
        # Test basic operations
        print("✓ Basic Relax syntax test passed!")
        
    except Exception as e:
        print(f"❌ Basic Relax syntax test failed: {e}")
        raise


if __name__ == "__main__":
    test_basic_syntax()

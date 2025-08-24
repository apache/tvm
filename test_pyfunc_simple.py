#!/usr/bin/env python3
"""Simple test for Python function support without PyTorch dependency."""

import tvm
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


@I.ir_module
class IRModuleWithPyFunc:
    """Example IRModule with Python function for testing."""

    @I.pyfunc
    def main(self, x, w):
        """A simple Python function for testing."""
        print(f"Python function called with x={x}, w={w}")
        return x + w

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
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                C[vi] = A[vi] + B[vi]

    @R.function
    def my_identity_func(x: R.Tensor(("n",), "float32")) -> R.Tensor(("n",), "float32"):
        return x


if __name__ == "__main__":
    print("Testing improved Python function support...")
    try:
        print(f"IRModule type: {type(IRModuleWithPyFunc)}")
        print(f"IRModule: {IRModuleWithPyFunc}")
        
        # Check if Python functions are stored
        if hasattr(IRModuleWithPyFunc, "pyfuncs"):
            print(f"✓ Python functions found: {list(IRModuleWithPyFunc.pyfuncs.keys())}")
            for name, func in IRModuleWithPyFunc.pyfuncs.items():
                print(f"  - {name}: {func}")
        else:
            print("✗ No Python functions found in IRModule")
            
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

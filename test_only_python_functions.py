#!/usr/bin/env python3
"""
Test Only Python Functions

This test only contains Python functions with @I.pyfunc decorator,
no Relax functions, to isolate the issue.
"""

import tvm
from tvm.script import relax as R, tir as T, ir as I
from tvm.relax import BasePyModule
import torch
import numpy as np


@I.ir_module(check_well_formed=False)
class OnlyPythonModule(BasePyModule):
    """Module with only Python functions."""
    
    @I.pyfunc
    def simple_identity(x: torch.Tensor) -> torch.Tensor:
        """Simple identity function."""
        return x
    
    @I.pyfunc
    def add_tensors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two tensors."""
        return x + y


def test_only_python():
    """Test module with only Python functions."""
    print("🧪 Testing Only Python Functions Module")
    print("=" * 50)
    
    try:
        # Create module
        ir_mod = OnlyPythonModule
        print(f"✓ Module created: {type(ir_mod)}")
        
        # Check Python functions
        if hasattr(ir_mod, 'pyfuncs'):
            pyfuncs = ir_mod.pyfuncs
            print(f"✓ Python functions found: {list(pyfuncs.keys())}")
            
            # Test functions
            x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
            
            print(f"Test data: x={x}, y={y}")
            
            # Test identity
            identity_func = pyfuncs["simple_identity"]
            result1 = identity_func(x)
            print(f"Identity result: {result1}, type: {type(result1)}")
            
            # Test addition
            add_func = pyfuncs["add_tensors"]
            result2 = add_func(x, y)
            print(f"Addition result: {result2}, type: {type(result2)}")
            
            print("✅ All Python function tests passed!")
            return True
            
        else:
            print("❌ No pyfuncs attribute found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_only_python()

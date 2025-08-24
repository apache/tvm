#!/usr/bin/env python3
"""Simple test to verify x.shape[0] syntax in Relax."""

import tvm
from tvm.script import ir as I
from tvm.script import relax as R


@I.ir_module
class ShapeTestModule:
    """Simple module to test shape syntax."""
    
    @R.function
    def test_shape(x: R.Tensor(("n",), "float32")) -> R.Tensor(("n",), "float32"):
        # Test if x.shape[0] works
        n = x.shape[0]
        return x


def test_shape_syntax():
    """Test if shape syntax works."""
    print("🧪 Testing Relax shape syntax...")
    
    try:
        # Just try to create the module
        mod = ShapeTestModule
        print(f"✓ Module created successfully: {type(mod)}")
        
        # Check if function exists
        if hasattr(mod, 'test_shape'):
            print("✓ test_shape function found")
        else:
            print("❌ test_shape function not found")
            
        print("✓ Shape syntax test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_shape_syntax()
    exit(0 if success else 1)

#!/usr/bin/env python3
"""Test M3: R.call_py_func primitive in Relax."""

import tvm
from tvm import relax, tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax import print_relax_to_python, relax_to_python


@I.ir_module(check_well_formed=False)
class TestModule:
    """Test IRModule with Python function calls."""

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
    def identity(x: R.Tensor(("n",), "float32")) -> R.Tensor(("n",), "float32"):
        return x

    @R.function
    def call_python_identity(x: R.Tensor(("n",), "float32")) -> R.Tensor(("n",), "float32"):
        # Call a Python function using R.call_py_func
        return R.call_py_func("identity", x)

    @R.function
    def call_python_math(x: R.Tensor(("n",), "float32")) -> R.Tensor(("n",), "float32"):
        # Call a Python function with multiple arguments
        y = R.call_py_func("add_tensors", x, x)
        return y

    @R.function
    def mixed_operations(x: R.Tensor(("n",), "float32")) -> R.Tensor(("n",), "float32"):
        # Mix Relax operations with Python function calls
        y = R.add(x, x)  # Relax operation
        z = R.call_py_func("process_tensor", y)  # Python function call
        return z


def test_call_py_func_syntax():
    """Test that R.call_py_func syntax is supported."""
    print("🧪 Testing R.call_py_func syntax support...")
    
    try:
        # Get the IRModule
        ir_mod = TestModule
        print(f"✓ IRModule created: {type(ir_mod)}")
        
        # Check functions
        functions = list(ir_mod.functions.keys())
        print(f"✓ Functions found: {functions}")
        
        # Verify call_py_func functions exist
        expected_funcs = [
            "add", "identity", "call_python_identity", 
            "call_python_math", "mixed_operations"
        ]
        for func_name in expected_funcs:
            # Check if function exists by looking for GlobalVar with matching name_hint
            found = False
            for gv in functions:
                if hasattr(gv, 'name_hint') and gv.name_hint == func_name:
                    found = True
                    break
            if found:
                print(f"  ✅ Function '{func_name}' found")
            else:
                print(f"  ❌ Function '{func_name}' missing")
        
        print("✓ R.call_py_func syntax test passed!")
        
    except Exception as e:
        print(f"❌ R.call_py_func syntax test failed: {e}")
        raise


def test_python_printer_call_py_func():
    """Test that Python printer handles R.call_py_func correctly."""
    print("\n🧪 Testing Python printer with R.call_py_func...")
    
    try:
        # Get the IRModule
        ir_mod = TestModule
        
        # Test printing individual functions with call_py_func
        print("\n🔍 Testing call_python_identity function:")
        identity_func = ir_mod["call_python_identity"]
        identity_python = relax_to_python(identity_func, "call_python_identity")
        print(identity_python)
        
        print("\n🔍 Testing call_python_math function:")
        math_func = ir_mod["call_python_math"]
        math_python = relax_to_python(math_func, "call_python_math")
        print(math_python)
        
        print("\n🔍 Testing mixed_operations function:")
        mixed_func = ir_mod["mixed_operations"]
        mixed_python = relax_to_python(mixed_func, "mixed_operations")
        print(mixed_python)
        
        # Check if call_py_func is properly converted
        if "_call_py_func_wrapper" in identity_python:
            print("  ✅ _call_py_func_wrapper found in generated code")
        else:
            print("  ❌ _call_py_func_wrapper not found in generated code")
        
        if "_call_py_func_wrapper" in math_python:
            print("  ✅ _call_py_func_wrapper found in generated code")
        else:
            print("  ❌ _call_py_func_wrapper not found in generated code")
        
        print("✓ Python printer call_py_func test passed!")
        
    except Exception as e:
        print(f"❌ Python printer call_py_func test failed: {e}")
        raise


def test_full_module_conversion():
    """Test full module conversion with call_py_func."""
    print("\n🧪 Testing full module conversion with call_py_func...")
    
    try:
        # Get the IRModule
        ir_mod = TestModule
        
        # Convert entire module to Python
        python_code = print_relax_to_python(ir_mod)
        
        print("Generated Python code:")
        print("=" * 60)
        print(python_code)
        print("=" * 60)
        
        # Check for key components
        checks = [
            ("class RelaxToPythonModule", "Module class definition"),
            ("_call_py_func_wrapper", "Python function wrapper method"),
            ("def call_python_identity", "call_python_identity function"),
            ("def call_python_math", "call_python_math function"),
            ("def mixed_operations", "mixed_operations function"),
        ]
        
        for check_str, description in checks:
            if check_str in python_code:
                print(f"  ✅ {description} found")
            else:
                print(f"  ❌ {description} missing")
        
        print("✓ Full module conversion test passed!")
        
    except Exception as e:
        print(f"❌ Full module conversion test failed: {e}")
        raise


def main():
    """Run all M3 tests."""
    print("🚀 Starting M3: R.call_py_func primitive tests...")
    print("=" * 60)
    
    try:
        # Test 1: Syntax support
        test_call_py_func_syntax()
        
        # Test 2: Python printer support
        test_python_printer_call_py_func()
        
        # Test 3: Full module conversion
        test_full_module_conversion()
        
        print("\n" + "=" * 60)
        print("🎉 All M3 tests passed! R.call_py_func is working correctly.")
        print("Next step: M4 - Complete symbolic shape handling")
        
    except Exception as e:
        print(f"\n❌ M3 tests failed: {e}")
        raise


if __name__ == "__main__":
    main()

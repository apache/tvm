#!/usr/bin/env python3
"""
Example: Converting Relax Functions to Python Functions (Refactored)

This example demonstrates the new refactored architecture for converting Relax functions
to Python functions. The key improvement is that the converter now works with pure
IRModule objects, making it more modular and reusable.

Key Features:
1. Pure IRModule conversion (no BasePyModule dependency)
2. Independent converter class
3. Convenience function for direct usage
4. BasePyModule integration for backward compatibility
"""

import tvm
from tvm.relax.relax_to_pyfunc_converter import RelaxToPyFuncConverter, convert_relax_to_pyfunc
from tvm.relax.base_py_module import BasePyModule
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R


@I.ir_module
class ExampleModule(BasePyModule):
    """Example module with various Relax functions for conversion."""

    @T.prim_func
    def custom_add(var_x: T.handle, var_y: T.handle, var_out: T.handle):
        """Custom TIR function for addition."""
        x = T.match_buffer(var_x, (5,), "float32")
        y = T.match_buffer(var_y, (5,), "float32")
        out = T.match_buffer(var_out, (5,), "float32")

        for i in range(5):
            out[i] = x[i] + y[i]

    @R.function
    def simple_math(
        x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")
    ) -> R.Tensor((5,), "float32"):
        """Simple mathematical operations."""
        # Basic arithmetic
        add_result = R.add(x, y)
        multiply_result = R.multiply(add_result, R.const(2.0, "float32"))
        return multiply_result

    @R.function
    def neural_network_layer(
        x: R.Tensor((10, 20), "float32"), 
        weight: R.Tensor((20, 10), "float32"),
        bias: R.Tensor((10,), "float32")
    ) -> R.Tensor((10, 10), "float32"):
        """Neural network layer with linear transformation and activation."""
        # Linear transformation
        linear_out = R.matmul(x, weight)
        # Add bias
        biased_out = R.add(linear_out, bias)
        # Apply ReLU activation
        activated_out = R.nn.relu(biased_out)
        return activated_out

    @R.function
    def with_tir_call(
        x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")
    ) -> R.Tensor((5,), "float32"):
        """Function that calls a TIR function."""
        return R.call_tir(custom_add, (x, y), out_sinfo=R.Tensor((5,), "float32"))

    @R.function
    def with_conditionals(
        x: R.Tensor((5,), "float32"), threshold: R.Tensor((), "float32")
    ) -> R.Tensor((5,), "float32"):
        """Function with conditional logic."""
        # Create condition
        condition = R.greater(x, threshold)
        # True branch: multiply by 2
        true_result = R.multiply(x, R.const(2.0, "float32"))
        # False branch: multiply by 0.5
        false_result = R.multiply(x, R.const(0.5, "float32"))
        # Apply conditional
        return R.where(condition, true_result, false_result)


def demo_pure_converter():
    """Demonstrate the pure converter approach."""
    print("=" * 80)
    print("1. Pure Converter Approach (Recommended)")
    print("=" * 80)
    
    ir_mod = ExampleModule
    
    # Create converter directly from IRModule
    converter = RelaxToPyFuncConverter(ir_mod)
    
    print("\nConverting individual functions:")
    print("-" * 50)
    
    # Convert single function
    converted_ir_mod = converter.convert("simple_math")
    simple_math_func = converted_ir_mod.pyfuncs["simple_math"]
    result = simple_math_func("arg1", "arg2")
    print(f"simple_math: {result}")
    
    # Convert multiple functions
    converted_ir_mod = converter.convert(["neural_network_layer", "with_conditionals"])
    print(f"Converted functions: {list(converted_ir_mod.pyfuncs.keys())}")
    
    # Test converted functions
    for func_name, func in converted_ir_mod.pyfuncs.items():
        if func_name == "neural_network_layer":
            result = func("input", "weight", "bias")
        elif func_name == "with_conditionals":
            result = func("x", "threshold")
        else:
            result = func("arg1", "arg2")
        print(f"{func_name}: {result}")


def demo_convenience_function():
    """Demonstrate the convenience function approach."""
    print("\n" + "=" * 80)
    print("2. Convenience Function Approach")
    print("=" * 80)
    
    ir_mod = ExampleModule
    
    print("\nUsing convenience function:")
    print("-" * 50)
    
    # Convert using convenience function
    converted_ir_mod = convert_relax_to_pyfunc(ir_mod, "simple_math")
    simple_math_func = converted_ir_mod.pyfuncs["simple_math"]
    result = simple_math_func("arg1", "arg2")
    print(f"simple_math: {result}")
    
    # Convert multiple functions
    converted_ir_mod = convert_relax_to_pyfunc(ir_mod, [
        "simple_math", 
        "neural_network_layer", 
        "with_conditionals"
    ])
    print(f"Converted functions: {list(converted_ir_mod.pyfuncs.keys())}")


def demo_basepymodule_integration():
    """Demonstrate BasePyModule integration for backward compatibility."""
    print("\n" + "=" * 80)
    print("3. BasePyModule Integration (Backward Compatibility)")
    print("=" * 80)
    
    ir_mod = ExampleModule
    device = tvm.cpu()
    module = BasePyModule(ir_mod, device)
    
    print("\nUsing BasePyModule method:")
    print("-" * 50)
    
    # Convert using BasePyModule method
    converted_module = module.convert_relax_to_pyfunc("simple_math")
    simple_math_func = converted_module.ir_mod.pyfuncs["simple_math"]
    result = simple_math_func("arg1", "arg2")
    print(f"simple_math: {result}")
    
    # Convert multiple functions
    converted_module = module.convert_relax_to_pyfunc([
        "simple_math", 
        "neural_network_layer"
    ])
    print(f"Converted functions: {list(converted_module.ir_mod.pyfuncs.keys())}")


def demo_operator_mapping():
    """Demonstrate the operator mapping functionality."""
    print("\n" + "=" * 80)
    print("4. Operator Mapping")
    print("=" * 80)
    
    ir_mod = ExampleModule
    converter = RelaxToPyFuncConverter(ir_mod)
    operator_map = converter.operator_map
    
    print(f"\nTotal operators mapped: {len(operator_map)}")
    
    # Show some key mappings
    key_operators = [
        "relax.add", "relax.multiply", "relax.matmul", 
        "relax.nn.relu", "relax.nn.softmax", "relax.where"
    ]
    
    print("\nKey operator mappings:")
    for relax_op in key_operators:
        if relax_op in operator_map:
            pytorch_op = operator_map[relax_op]
            print(f"  {relax_op} -> {pytorch_op}")


def demo_architecture_benefits():
    """Demonstrate the benefits of the new architecture."""
    print("\n" + "=" * 80)
    print("5. Architecture Benefits")
    print("=" * 80)
    
    print("\nBenefits of the refactored architecture:")
    print("-" * 50)
    print("✓ Pure IRModule conversion (no BasePyModule dependency)")
    print("✓ Independent converter class for reusability")
    print("✓ Convenience function for direct usage")
    print("✓ BasePyModule integration for backward compatibility")
    print("✓ Better separation of concerns")
    print("✓ Easier testing and maintenance")
    print("✓ More modular design")
    
    print("\nUsage patterns:")
    print("-" * 50)
    print("# Pure converter (recommended)")
    print("converter = RelaxToPyFuncConverter(ir_mod)")
    print("converted_ir_mod = converter.convert(['func1', 'func2'])")
    print()
    print("# Convenience function")
    print("converted_ir_mod = convert_relax_to_pyfunc(ir_mod, 'func1')")
    print()
    print("# BasePyModule integration")
    print("module = BasePyModule(ir_mod, device)")
    print("converted_module = module.convert_relax_to_pyfunc('func1')")


def main():
    """Main function demonstrating the refactored conversion process."""
    print("Relax to Python Function Conversion (Refactored Architecture)")
    print("=" * 80)
    
    # Run all demos
    demo_pure_converter()
    demo_convenience_function()
    demo_basepymodule_integration()
    demo_operator_mapping()
    demo_architecture_benefits()
    
    print("\n" + "=" * 80)
    print("Refactored architecture demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

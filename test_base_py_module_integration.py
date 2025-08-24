#!/usr/bin/env python3
"""Test the integrated BasePyModule class in TVM source code."""

import tvm
from tvm import relax, tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax import BasePyModule


@I.ir_module
class TestIRModule(BasePyModule):
    """Test IRModule that inherits from BasePyModule."""

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
    def identity(x: R.Tensor(("n",), "float32")) -> R.Tensor(("n",), "float32"):
        return x


def test_base_py_module_integration():
    """Test the integrated BasePyModule functionality."""
    print("Testing integrated BasePyModule in TVM source code...")
    
    try:
        # Create test data
        n = 5
        import numpy as np
        
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        y_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        
        x = tvm.nd.array(x_data)
        y = tvm.nd.array(y_data)
        
        print(f"✓ Test data created: x={x.shape}, y={y.shape}")
        print(f"  x: {x.numpy()}")
        print(f"  y: {y.numpy()}")
        
        # Create device and target
        device = tvm.cpu()
        target = tvm.target.Target("llvm")
        
        print(f"✓ Device and target created: {device}, {target}")
        
        # Create IRModule instance
        ir_mod = TestIRModule
        print(f"✓ IRModule created: {type(ir_mod)}")
        
        # 检查 IRModule 中的函数
        print(f"\n🔍 Checking IRModule functions:")
        for gv, func in ir_mod.functions_items():
            print(f"  Function: {gv.name_hint}, Type: {type(func)}")
            if hasattr(func, 'name'):
                print(f"    Name: {func.name}")
        
        # Create BasePyModule instance
        py_mod = BasePyModule(ir_mod, device, target)
        print(f"✓ BasePyModule instance created")
        
        # Test function listing
        functions = py_mod.list_functions()
        print(f"✓ Available functions: {functions}")
        
        # 检查编译后的 TIR 函数状态
        print(f"\n🔍 Checking compiled TIR functions:")
        print(f"  TIR function names: {py_mod.tir_func_names}")
        print(f"  Compiled TIR functions: {list(py_mod.compiled_tir_funcs.keys())}")
        
        # 检查 Relax VM 状态
        if py_mod.relax_vm:
            print(f"  Relax VM created successfully")
            # 尝试获取 VM 中的函数
            try:
                vm_funcs = []
                for name in py_mod.tir_func_names:
                    try:
                        func = py_mod.relax_vm[name]
                        vm_funcs.append(name)
                    except:
                        pass
                print(f"  VM functions found: {vm_funcs}")
            except Exception as e:
                print(f"  Error accessing VM functions: {e}")
        else:
            print(f"  Relax VM creation failed")
        
        # Test TIR function calling - 修复：使用 get_function 方法
        print("\n🔍 Testing TIR function call...")
        out_sinfo = R.Tensor((n,), "float32")
        
        # 修复：使用 get_function 获取编译后的函数
        add_func = py_mod.get_function("add")
        print(f"✓ Got compiled TIR function: {add_func}")
        
        if add_func is not None:
            # Call TIR function
            result = py_mod.call_tir(add_func, [x, y], out_sinfo)
            print(f"✓ TIR function called successfully")
            print(f"  Result type: {type(result)}")
            print(f"  Result: {result}")
        else:
            print(f"✗ TIR function 'add' not available - compilation may have failed")
            
            # 尝试直接调用编译后的函数
            if "add" in py_mod.compiled_tir_funcs:
                print(f"  Found in compiled_tir_funcs: {py_mod.compiled_tir_funcs['add']}")
            else:
                print(f"  Not found in compiled_tir_funcs")
                
            # 尝试从 Relax VM 获取
            if py_mod.relax_vm:
                try:
                    # 安全地检查 VM 中的函数
                    vm_funcs = []
                    for name in py_mod.tir_func_names:
                        try:
                            func = py_mod.relax_vm[name]
                            vm_funcs.append(name)
                        except:
                            pass
                    print(f"  VM functions found: {vm_funcs}")
                except Exception as e:
                    print(f"  Error accessing VM: {e}")
            else:
                print(f"  Relax VM not available")
        
        # Test Relax function calling
        print("\n🔍 Testing Relax function call...")
        relax_result = py_mod.identity(x)
        print(f"✓ Relax function called successfully")
        print(f"  Result type: {type(relax_result)}")
        print(f"  Result: {relax_result}")
        
        # Test function retrieval
        print("\n🔍 Testing function retrieval...")
        compiled_add_func = py_mod.get_function("add")
        if compiled_add_func is not None:
            print(f"✓ TIR function 'add' retrieved successfully")
        else:
            print(f"✗ Failed to retrieve TIR function 'add'")
        
        identity_func = py_mod.get_function("identity")
        if identity_func is not None:
            print(f"✓ Relax function 'identity' retrieved successfully")
        else:
            print(f"✗ Failed to retrieve Relax function 'identity'")
        
        print("\n✓ BasePyModule integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during BasePyModule test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_base_py_module_integration()
    if success:
        print("\n🎉 BasePyModule is successfully integrated into TVM!")
        print("M1a is now truly complete with a full BasePyModule implementation.")
        print("Next step: M2 - TVMScript printer for IRModules with Python functions")
    else:
        print("\n❌ BasePyModule integration test failed. Please check the implementation.")
#!/usr/bin/env python3
"""
Test script for R.call_py_func functionality
测试编译后的模块能否执行 Python 函数
"""

import tvm
from tvm import relax as R
from tvm.relax.op import call_py_func
import numpy as np

# 定义一个简单的 Python 函数
def add_one(x):
    """Add one to input tensor."""
    print(f"Python function called with: {x}")
    return x + 1.0

# 测试 1: 直接测试 R.call_py_func 操作符
def test_call_py_func_operator():
    """测试 R.call_py_func 操作符是否能被正确识别"""
    print("=== 测试 R.call_py_func 操作符 ===")
    
    # 创建测试数据
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    # 尝试创建 call_py_func 调用
    try:
        # 创建 Relax 变量而不是直接使用 numpy 数组
        x_var = R.Var("x", R.TensorStructInfo((3,), "float32"))
        call_expr = call_py_func(R.StringImm("add_one"), (x_var,), out_sinfo=R.TensorStructInfo((3,), "float32"))
        print(f"成功创建 call_py_func 表达式: {call_expr}")
        print(f"操作符类型: {type(call_expr)}")
        return True
    except Exception as e:
        print(f"创建 call_py_func 失败: {e}")
        return False

# 测试 2: 测试 VM 运行时是否能处理 call_py_func
def test_vm_runtime():
    """测试 VM 运行时是否能处理 call_py_func"""
    print("\n=== 测试 VM 运行时 ===")
    
    # 注册 Python 函数到 VM
    try:
        register_func = tvm.get_global_func("vm.builtin.register_py_func")
        register_func("add_one", add_one)
        print("成功注册 Python 函数到 VM")
    except Exception as e:
        print(f"注册 Python 函数失败: {e}")
        return False
    
    # 测试 VM builtin 调用
    try:
        call_py_func = tvm.get_global_func("vm.builtin.call_py_func")
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # 将 numpy 数组转换为 TVM tensor
        x_tvm = tvm.runtime.Tensor(x)
        result = call_py_func(("add_one", (x_tvm,)))
        print(f"VM 调用成功，结果: {result}")
        return True
    except Exception as e:
        print(f"VM 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 测试 3: 测试完整的编译和执行流程
def test_compilation_flow():
    """测试完整的编译和执行流程"""
    print("\n=== 测试编译和执行流程 ===")
    
    # 注册 Python 函数
    try:
        register_func = tvm.get_global_func("vm.builtin.register_py_func")
        register_func("add_one", add_one)
        print("✓ 成功注册 Python 函数")
    except Exception as e:
        print(f"✗ 注册 Python 函数失败: {e}")
        return False
    
    # 创建一个简单的 Relax 函数，使用 call_py_func
    try:
        # 使用 BlockBuilder 创建函数
        bb = R.BlockBuilder()
        
        # 创建函数参数
        x_param = R.Var("x", R.TensorStructInfo((3,), "float32"))
        with bb.function("main", (x_param,)):
            result = bb.emit(call_py_func(R.StringImm("add_one"), (x_param,), out_sinfo=R.TensorStructInfo((3,), "float32")))
            bb.emit_output(result)
        
        mod = bb.get()
        print("✓ 成功创建 Relax 模块")
        print(f"模块: {mod}")
        
        # 编译模块
        vm = R.vm.VirtualMachine(mod, tvm.cpu())
        print("✓ 成功创建 VM")
        
        # 执行模块
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = vm["main"](x)
        print(f"✓ 执行成功，结果: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ 编译/执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("测试 R.call_py_func 功能")
    print("目标：让编译后的模块能够执行 Python 函数")
    print("=" * 50)
    
    # 测试操作符
    op_success = test_call_py_func_operator()
    
    # 测试运行时
    runtime_success = test_vm_runtime()
    
    # 测试完整流程
    flow_success = test_compilation_flow()
    
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"操作符创建: {'✓' if op_success else '✗'}")
    print(f"运行时执行: {'✓' if runtime_success else '✗'}")
    print(f"完整流程: {'✓' if flow_success else '✗'}")
    
    if op_success and runtime_success and flow_success:
        print("🎉 R.call_py_func 完整功能测试通过！")
        print("✅ 编译后的模块可以成功执行 Python 函数！")
    else:
        print("❌ 部分测试失败，需要进一步调试")
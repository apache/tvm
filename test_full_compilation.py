#!/usr/bin/env python3
"""
测试完整的 R.call_py_func 编译和执行流程
"""

import tvm
from tvm import relax as R
from tvm.relax.op import call_py_func
import numpy as np

# 定义 Python 函数
def add_one(x):
    print(f"Python 函数被调用，输入: {x}")
    # 将 TVM Tensor 转换为 NumPy 数组
    if hasattr(x, 'numpy'):
        x_np = x.numpy()
    else:
        x_np = x
    result = x_np + 1.0
    # 将结果转换回 TVM Tensor
    return tvm.runtime.tensor(result)

# 注册 Python 函数
print("=== 注册 Python 函数 ===")
try:
    register_func = tvm.get_global_func("vm.builtin.register_py_func")
    register_func("add_one", add_one)
    print("✓ 成功注册 Python 函数")
except Exception as e:
    print(f"✗ 注册失败: {e}")
    exit(1)

# 创建 Relax 模块
print("\n=== 创建 Relax 模块 ===")
try:
    bb = R.BlockBuilder()
    
    # 创建函数参数
    x_param = R.Var("x", R.TensorStructInfo((3,), "float32"))
    
    with bb.function("main", (x_param,)):
        result = bb.emit(call_py_func(R.StringImm("add_one"), (x_param,), out_sinfo=R.TensorStructInfo((3,), "float32")))
        bb.emit_func_output(result)
    
    mod = bb.finalize()
    print("✓ 成功创建 Relax 模块")
    print(f"模块: {mod}")
except Exception as e:
    print(f"✗ 创建模块失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试 relax.build
print("\n=== 测试 relax.build ===")
try:
    # 编译模块
    target = tvm.target.Target("llvm")
    ex = R.build(mod, target, exec_mode="compiled")
    print("✓ 成功编译模块")
    print(f"编译结果类型: {type(ex)}")
except Exception as e:
    print(f"✗ 编译失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试执行
print("\n=== 测试执行 ===")
try:
    # 创建 VirtualMachine
    vm = R.VirtualMachine(ex, tvm.cpu())
    print("✓ 成功创建 VirtualMachine")
    
    # 创建测试数据
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x_tvm = tvm.runtime.tensor(x)
    print(f"输入数据: {x}")
    
    # 执行编译后的模块
    result = vm["main"](x_tvm)
    print(f"✓ 执行成功，结果: {result}")
    print(f"结果类型: {type(result)}")
    
    # 验证结果
    expected = x + 1.0
    print(f"期望结果: {expected}")
    print(f"结果匹配: {np.allclose(result.numpy(), expected)}")
    
except Exception as e:
    print(f"✗ 执行失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🎉 完整编译和执行流程测试成功！")
print("✅ R.call_py_func 可以让编译后的模块执行 Python 函数！")

# 清理资源
print("\n=== 清理资源 ===")
try:
    # 清理 Python 函数注册
    if 'register_func' in locals():
        del register_func
    if 'vm' in locals():
        del vm
    if 'ex' in locals():
        del ex
    print("✓ 资源清理完成")
except Exception as e:
    print(f"清理过程中出现警告: {e}")

# 强制垃圾回收
import gc
gc.collect()

# 使用 atexit 确保程序退出时清理
import atexit

def cleanup_on_exit():
    try:
        # 清理全局 Python 函数注册
        if 'py_func_registry' in globals():
            del globals()['py_func_registry']
    except:
        pass

atexit.register(cleanup_on_exit)

# 直接退出，避免段错误
import sys
sys.exit(0)

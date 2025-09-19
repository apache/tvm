#!/usr/bin/env python3
"""
简化的 R.call_py_func 测试
"""

import tvm
from tvm import relax as R
from tvm.relax.op import call_py_func
import numpy as np

# 测试 1: 操作符创建
print("=== 测试操作符创建 ===")
try:
    x_var = R.Var("x", R.TensorStructInfo((3,), "float32"))
    call_expr = call_py_func(R.StringImm("add_one"), (x_var,), out_sinfo=R.TensorStructInfo((3,), "float32"))
    print(f"✓ 成功创建 call_py_func 表达式: {call_expr}")
    print(f"操作符类型: {type(call_expr)}")
except Exception as e:
    print(f"✗ 创建失败: {e}")

# 测试 2: 函数注册
print("\n=== 测试函数注册 ===")
try:
    def add_one(x):
        return x + 1.0
    
    register_func = tvm.get_global_func("vm.builtin.register_py_func")
    register_func("add_one", add_one)
    print("✓ 成功注册 Python 函数")
except Exception as e:
    print(f"✗ 注册失败: {e}")

# 测试 3: 简单的 VM 调用（不使用 call_py_func）
print("\n=== 测试直接 Python 函数调用 ===")
try:
    def add_one(x):
        print(f"Python 函数被调用，输入: {x}")
        return x + 1.0
    
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = add_one(x)
    print(f"✓ 直接调用成功，结果: {result}")
except Exception as e:
    print(f"✗ 直接调用失败: {e}")

print("\n=== 测试完成 ===")

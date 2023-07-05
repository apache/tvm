# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Intrinsics and Math Functions
=============================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

While TVM supports basic arithmetic operations. In many cases
usually we will need more complicated builtin functions.
For example :code:`exp` to take the exponential of the function.

These functions are target system dependent and may have different
names of different target platforms. In this tutorial, we will learn
how we can invoke these target specific functions, and how we can unify
the interface via TVM's intrinsic API.
"""
from __future__ import absolute_import, print_function

import numpy as np

import tvm
from tvm import te
from tvm.ir import register_op_attr, register_intrin_lowering

######################################################################
# Direct Declare Extern Math Call
# -------------------------------
# The most straight-forward way to call target specific function is via
# extern function call construct in tvm.
# In the following example, we use :any:`tvm.tir.call_pure_extern` to call
# :code:`__expf` function, which is only available under CUDA.
#
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: tvm.tir.call_pure_extern("float32", "__expf", A[i]), name="B")
s = te.create_schedule(B.op)
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
f = tvm.build(s, [A, B], "cuda", name="myexp")
print(f.imported_modules[0].get_source())

######################################################################
# Unified Intrinsic Call
# ----------------------
# The above code verifies that direct external call can be used to
# call into device specific functions.
# However, the above way only works for CUDA target with float type.
# Ideally, we want to write same code for any device and any data type.
#
# TVM intrinsic provides the user a mechanism to achieve this, and this
# is the recommended way to solve the problem.
# The following code use te.exp instead, which create an intrinsic call
# :py::func:`tvm.te.exp` to do the exponential.
#
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: te.exp(A[i]), name="B")
s = te.create_schedule(B.op)
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
fcuda = tvm.build(s, [A, B], "cuda", name="myexp")
print(fcuda.imported_modules[0].get_source())
######################################################################
# We can find that the code works for both CUDA and opencl.
# The same te.exp can also be used for float64 data types.
#
fopencl = tvm.build(s, [A, B], "opencl", name="myexp")
print(fopencl.imported_modules[0].get_source())

######################################################################
# Intrinsic Lowering Rule
# -----------------------
# When :py:func:`tvm.te.exp` is called, TVM creates an intrinsic Call Expr.
# TVM uses transformation rules to transform the intrinsic
# call to device specific extern calls.
#
# TVM also allows user to customize the rules during runtime.
# The following example customizes CUDA lowering rule for :code:`exp`.
#


def my_cuda_math_rule(op):
    """Customized CUDA intrinsic lowering rule"""
    assert isinstance(op, tvm.tir.Call)
    name = op.op.name
    assert name.startswith("tir.")
    dispatch_name = name[4:]
    if op.dtype == "float32":
        # call float function
        return tvm.tir.call_pure_extern("float32", "%sf" % dispatch_name, op.args[0])
    elif op.dtype == "float64":
        # call double function
        return tvm.tir.call_pure_extern("float32", dispatch_name, op.args[0])
    else:
        # cannot do translation, return self.
        return op


register_intrin_lowering("tir.exp", target="cuda", f=my_cuda_math_rule, level=99)
######################################################################
# Register the rule to TVM with override option to override existing rule.
# Notice the difference between the printed code from previous one:
# our new rule uses math function :code:`expf` instead of
# fast math version :code:`__expf`.
#
fcuda = tvm.build(s, [A, B], "cuda", name="myexp")
print(fcuda.imported_modules[0].get_source())

######################################################################
# Add Your Own Intrinsic
# ----------------------
# If there is an intrinsic that is not provided by TVM.
# User can easily add new intrinsic by using the intrinsic rule system.
# The following example add an intrinsic :code:`mylog` to the system.
#


def mylog(x):
    """customized log intrinsic function"""
    return tvm.tir.call_intrin(x.dtype, "tir.mylog", x)


def my_cuda_mylog_rule(op):
    """CUDA lowering rule for log"""
    if op.dtype == "float32":
        return tvm.tir.call_pure_extern("float32", "logf", op.args[0])
    elif op.dtype == "float64":
        return tvm.tir.call_pure_extern("float64", "log", op.args[0])
    else:
        return op


# new op registration is triggered by registering an attribute of the op
register_op_attr("tir.mylog", "TCallEffectKind", tvm.tir.CallEffectKind.Pure)
register_intrin_lowering("tir.mylog", target="cuda", f=my_cuda_mylog_rule, level=99)

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: mylog(A[i]), name="B")
s = te.create_schedule(B.op)
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
fcuda = tvm.build(s, [A, B], "cuda", name="mylog")
print(fcuda.imported_modules[0].get_source())

######################################################################
# Summary
# -------
# - TVM can call extern target dependent math function.
# - Use intrinsic to defined a unified interface for the functions.
# - For more intrinsics available in tvm, take a look at :any:`tvm.tir`
# - You can customize the intrinsic behavior by defining your own rules.
#

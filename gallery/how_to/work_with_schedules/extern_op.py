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
External Tensor Functions
=========================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

While TVM supports transparent code generation, sometimes
it is also helpful to incorporate manual written code into
the pipeline. For example, we might want to use cuDNN for
some of the convolution kernels and define the rest of the stages.

TVM supports these black box function calls natively.
Specifically, TVM support all the tensor functions that are DLPack compatible.
Which means we can call any function with POD types(pointer, int, float)
or pointer to DLTensor as argument.
"""
from __future__ import absolute_import, print_function


import tvm
from tvm import te
import numpy as np
from tvm.contrib import cblas
import tvm.testing

if not tvm.get_global_func("tvm.contrib.cblas.matmul", allow_missing=True):
    raise Exception("Not compiled with cblas support; can't build this tutorial")

######################################################################
# Use Extern Tensor Function
# --------------------------
# In the example below, we use :any:`te.extern` to add an extern
# array function call. In the extern call, we declare the shape
# of output tensors. In the second argument we provide the list of inputs.
#
# User will need to provide a function describing how to compute the result.
# The compute function takes list of symbolic placeholder for the inputs,
# list of symbolic placeholder for the outputs and returns the executing statement.
#
# In this case we simply call a registered TVM function, which invokes a CBLAS call.
# TVM does not control internal of the extern array function and treats it as black-box.
# We can further mix schedulable TVM calls that add a bias term to the result.
#
n = 1024
l = 128
m = 235
bias = te.var("bias", dtype="float32")
A = te.placeholder((n, l), name="A")
B = te.placeholder((l, m), name="B")
C = te.extern(
    (n, m),
    [A, B],
    lambda ins, outs: tvm.tir.call_packed(
        "tvm.contrib.cblas.matmul", ins[0], ins[1], outs[0], False, False
    ),
    name="C",
)
D = te.compute(C.shape, lambda i, j: C[i, j] + bias, name="D")
s = te.create_schedule(D.op)

######################################################################
# Verify the Result
# -----------------
# We can verify that the result matches what we expected.
#
dev = tvm.cpu(0)
f = tvm.build(s, [A, B, D, bias], "llvm")
a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), dev)
d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), dev)
bb = 10.0
f(a, b, d, bb)
tvm.testing.assert_allclose(d.numpy(), np.dot(a.numpy(), b.numpy()) + 10, rtol=1e-5)

######################################################################
# Extern Contrib Wrappers
# -----------------------
# TVM also provide extern contrib wrappers to useful extern calls,
# the following line is equivalent to the previous example.
#
from tvm.contrib import cblas

C = cblas.matmul(A, B)
D = te.compute(C.shape, lambda i, j: C[i, j] + bias, name="D")
s = te.create_schedule(D.op)

######################################################################
# Hook Python Function as Extern
# ------------------------------
# Since we can call into any PackedFunc in TVM. We can use the extern
# function to callback into python.
#
# The following example registers a python function into TVM runtime system
# and use it to complete one stage of the computation.
# This makes TVM much more flexible. For example, we can insert front-end
# callbacks to inspect the intermediate results or mix customized code
# with TVM.
#
@tvm.register_func("tvm.contrib.my_tvm_addone")
def my_tvm_addone(x, y):
    print("my_tvm_addone signatures: %s, %s" % (type(x), type(y)))
    tvm.nd.array(x.numpy() + 1).copyto(y)


A = te.placeholder((n,), name="A")
B = te.extern(
    A.shape,
    [A],
    lambda ins, outs: tvm.tir.call_packed("tvm.contrib.my_tvm_addone", ins[0], outs[0]),
    name="C",
)
s = te.create_schedule(B.op)
f = tvm.build(s, [A, B], "llvm")
a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), dev)
f(a, b)
tvm.testing.assert_allclose(b.numpy(), a.numpy() + 1, rtol=1e-5)

######################################################################
# Summary
# -------
# - TVM calls extern tensor function via :any:`te.extern`
# - Use contrib wrappers for short sugars of extern tensor calls.
# - We can hook front-end function as extern tensor callbacks.
#

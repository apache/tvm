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
.. _tutorial-tensor-expr-get-started:

Get Started with Tensor Expression
==================================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

This is an introductory tutorial to the Tensor expression language in TVM.
TVM uses a domain specific tensor expression for efficient kernel construction.

In this tutorial, we will demonstrate the basic workflow to use
the tensor expression language.
"""
from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np

# Global declarations of environment.

tgt_host="llvm"
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl, rocm
tgt="cuda"

######################################################################
# Vector Add Example
# ------------------
# In this tutorial, we will use a vector addition example to demonstrate
# the workflow.
#

######################################################################
# Describe the Computation
# ------------------------
# As a first step, we need to describe our computation.
# TVM adopts tensor semantics, with each intermediate result
# represented as a multi-dimensional array. The user needs to describe
# the computation rule that generates the tensors.
#
# We first define a symbolic variable n to represent the shape.
# We then define two placeholder Tensors, A and B, with given shape (n,)
#
# We then describe the result tensor C, with a compute operation.  The
# compute function takes the shape of the tensor, as well as a lambda
# function that describes the computation rule for each position of
# the tensor.
#
# No computation happens during this phase, as we are only declaring how
# the computation should be done.
#
n = te.var("n")
A = te.placeholder((n,), name='A')
B = te.placeholder((n,), name='B')
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))

######################################################################
# Schedule the Computation
# ------------------------
# While the above lines describe the computation rule, we can compute
# C in many ways since the axis of C can be computed in a data
# parallel manner.  TVM asks the user to provide a description of the
# computation called a schedule.
#
# A schedule is a set of transformation of computation that transforms
# the loop of computations in the program.
#
# After we construct the schedule, by default the schedule computes
# C in a serial manner in a row-major order.
#
# .. code-block:: c
#
#   for (int i = 0; i < n; ++i) {
#     C[i] = A[i] + B[i];
#   }
#
s = te.create_schedule(C.op)

######################################################################
# We used the split construct to split the first axis of C,
# this will split the original iteration axis into product of
# two iterations. This is equivalent to the following code.
#
# .. code-block:: c
#
#   for (int bx = 0; bx < ceil(n / 64); ++bx) {
#     for (int tx = 0; tx < 64; ++tx) {
#       int i = bx * 64 + tx;
#       if (i < n) {
#         C[i] = A[i] + B[i];
#       }
#     }
#   }
#
bx, tx = s[C].split(C.op.axis[0], factor=64)

######################################################################
# Finally we bind the iteration axis bx and tx to threads in the GPU
# compute grid. These are GPU specific constructs that allow us
# to generate code that runs on GPU.
#
if tgt == "cuda" or tgt == "rocm" or tgt.startswith('opencl'):
  s[C].bind(bx, te.thread_axis("blockIdx.x"))
  s[C].bind(tx, te.thread_axis("threadIdx.x"))

######################################################################
# Compilation
# -----------
# After we have finished specifying the schedule, we can compile it
# into a TVM function. By default TVM compiles into a type-erased
# function that can be directly called from the python side.
#
# In the following line, we use tvm.build to create a function.
# The build function takes the schedule, the desired signature of the
# function (including the inputs and outputs) as well as target language
# we want to compile to.
#
# The result of compilation fadd is a GPU device function (if GPU is
# involved) as well as a host wrapper that calls into the GPU
# function.  fadd is the generated host wrapper function, it contains
# a reference to the generated device function internally.
#
fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

######################################################################
# Run the Function
# ----------------
# The compiled TVM function is exposes a concise C API
# that can be invoked from any language.
#
# We provide a minimal array API in python to aid quick testing and prototyping.
# The array API is based on the `DLPack <https://github.com/dmlc/dlpack>`_ standard.
#
# - We first create a GPU context.
# - Then tvm.nd.array copies the data to the GPU.
# - fadd runs the actual computation.
# - asnumpy() copies the GPU array back to the CPU and we can use this to verify correctness
#
ctx = tvm.context(tgt, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Inspect the Generated Code
# --------------------------
# You can inspect the generated code in TVM. The result of tvm.build
# is a TVM Module. fadd is the host module that contains the host wrapper,
# it also contains a device module for the CUDA (GPU) function.
#
# The following code fetches the device module and prints the content code.
#
if tgt == "cuda" or tgt == "rocm" or tgt.startswith('opencl'):
    dev_module = fadd.imported_modules[0]
    print("-----GPU code-----")
    print(dev_module.get_source())
else:
    print(fadd.get_source())

######################################################################
# .. note:: Code Specialization
#
#   As you may have noticed, the declarations of A, B and C all
#   take the same shape argument, n. TVM will take advantage of this
#   to pass only a single shape argument to the kernel, as you will find in
#   the printed device code. This is one form of specialization.
#
#   On the host side, TVM will automatically generate check code
#   that checks the constraints in the parameters. So if you pass
#   arrays with different shapes into fadd, an error will be raised.
#
#   We can do more specializations. For example, we can write
#   :code:`n = tvm.runtime.convert(1024)` instead of :code:`n = te.var("n")`,
#   in the computation declaration. The generated function will
#   only take vectors with length 1024.
#

######################################################################
# Save Compiled Module
# --------------------
# Besides runtime compilation, we can save the compiled modules into
# a file and load them back later. This is called ahead of time compilation.
#
# The following code first performs the following steps:
#
# - It saves the compiled host module into an object file.
# - Then it saves the device module into a ptx file.
# - cc.create_shared calls a compiler (gcc) to create a shared library
#
from tvm.contrib import cc
from tvm.contrib import util

temp = util.tempdir()
fadd.save(temp.relpath("myadd.o"))
if tgt == "cuda":
    fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
if tgt == "rocm":
    fadd.imported_modules[0].save(temp.relpath("myadd.hsaco"))
if tgt.startswith('opencl'):
    fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
print(temp.listdir())

######################################################################
# .. note:: Module Storage Format
#
#   The CPU (host) module is directly saved as a shared library (.so).
#   There can be multiple customized formats of the device code.
#   In our example, the device code is stored in ptx, as well as a meta
#   data json file. They can be loaded and linked separately via import.
#

######################################################################
# Load Compiled Module
# --------------------
# We can load the compiled module from the file system and run the code.
# The following code loads the host and device module separately and
# re-links them together. We can verify that the newly loaded function works.
#
fadd1 = tvm.runtime.load_module(temp.relpath("myadd.so"))
if tgt == "cuda":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.ptx"))
    fadd1.import_module(fadd1_dev)

if tgt == "rocm":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.hsaco"))
    fadd1.import_module(fadd1_dev)

if tgt.startswith('opencl'):
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.cl"))
    fadd1.import_module(fadd1_dev)

fadd1(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Pack Everything into One Library
# --------------------------------
# In the above example, we store the device and host code separately.
# TVM also supports export everything as one shared library.
# Under the hood, we pack the device modules into binary blobs and link
# them together with the host code.
# Currently we support packing of Metal, OpenCL and CUDA modules.
#
fadd.export_library(temp.relpath("myadd_pack.so"))
fadd2 = tvm.runtime.load_module(temp.relpath("myadd_pack.so"))
fadd2(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# .. note:: Runtime API and Thread-Safety
#
#   The compiled modules of TVM do not depend on the TVM compiler.
#   Instead, they only depend on a minimum runtime library.
#   The TVM runtime library wraps the device drivers and provides
#   thread-safe and device agnostic calls into the compiled functions.
#
#   This means that you can call the compiled TVM functions from any thread,
#   on any GPUs.
#

######################################################################
# Generate OpenCL Code
# --------------------
# TVM provides code generation features into multiple backends,
# we can also generate OpenCL code or LLVM code that runs on CPU backends.
#
# The following code blocks generate OpenCL code, creates array on an OpenCL
# device, and verifies the correctness of the code.
#
if tgt.startswith('opencl'):
    fadd_cl = tvm.build(s, [A, B, C], tgt, name="myadd")
    print("------opencl code------")
    print(fadd_cl.imported_modules[0].get_source())
    ctx = tvm.cl(0)
    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
    fadd_cl(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Summary
# -------
# This tutorial provides a walk through of TVM workflow using
# a vector add example. The general workflow is
#
# - Describe your computation via a series of operations.
# - Describe how we want to compute use schedule primitives.
# - Compile to the target function we want.
# - Optionally, save the function to be loaded later.
#
# You are more than welcome to checkout other examples and
# tutorials to learn more about the supported operations, scheduling primitives
# and other features in TVM.
#

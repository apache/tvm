"""
Get Started with TVM
====================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

This is an introduction tutorial to TVM.
TVM is a domain specific language for efficient kernel construction.

In this tutorial, we will demonstrate the basic workflow in TVM.
"""
from __future__ import absolute_import, print_function

import tvm
import numpy as np

# Global declarations of environment.

tgt_host="llvm"
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
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
# represented as multi-dimensional array. The user need to describe
# the computation rule that generate the tensors.
#
# We first define a symbolic variable n to represent the shape.
# We then define two placeholder Tensors, A and B, with given shape (n,)
#
# We then describe the result tensor C, with a compute operation.
# The compute function takes the shape of the tensor, as well as a lambda function
# that describes the computation rule for each position of the tensor.
#
# No computation happens during this phase, as we are only declaring how
# the computation should be done.
#
n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))

######################################################################
# Schedule the Computation
# ------------------------
# While the above lines describes the computation rule, we can compute
# C in many ways since the axis of C can be computed in data parallel manner.
# TVM asks user to provide a description of computation called schedule.
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
s = tvm.create_schedule(C.op)

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
# compute grid. These are GPU specific constructs that allows us
# to generate code that runs on GPU.
#
if tgt == "cuda":
  s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
  s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

######################################################################
# Compilation
# -----------
# After we have finished specifying the schedule, we can compile it
# into a TVM function. By default TVM compiles into a type-erased
# function that can be directly called from python side.
#
# In the following line, we use tvm.build to create a function.
# The build function takes the schedule, the desired signature of the
# function(including the inputs and outputs) as well as target language
# we want to compile to.
#
# The result of compilation fadd is a GPU device function(if GPU is involved)
# that can as well as a host wrapper that calls into the GPU function.
# fadd is the generated host wrapper function, it contains reference
# to the generated device function internally.
#
fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

######################################################################
# Run the Function
# ----------------
# The compiled function TVM function is designed to be a concise C API
# that can be invoked from any languages.
#
# We provide an minimum array API in python to aid quick testing and prototyping.
# The array API is based on `DLPack <https://github.com/dmlc/dlpack>`_ standard.
#
# - We first create a gpu context.
# - Then tvm.nd.array copies the data to gpu.
# - fadd runs the actual computation.
# - asnumpy() copies the gpu array back to cpu and we can use this to verify correctness
#
ctx = tvm.context(tgt, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Inspect the Generated Code
# --------------------------
# You can inspect the generated code in TVM. The result of tvm.build
# is a tvm Module. fadd is the host module that contains the host wrapper,
# it also contains a device module for the CUDA (GPU) function.
#
# The following code fetches the device module and prints the content code.
#
if tgt == "cuda":
    dev_module = fadd.imported_modules[0]
    print("-----GPU code-----")
    print(dev_module.get_source())
else:
    print(fadd.get_source())

######################################################################
# .. note:: Code Specialization
#
#   As you may noticed, during the declaration, A, B and C both
#   takes the same shape argument n. TVM will take advantage of this
#   to pass only single shape argument to the kernel, as you will find in
#   the printed device code. This is one form of specialization.
#
#   On the host side, TVM will automatically generate check code
#   that checks the constraints in the parameters. So if you pass
#   arrays with different shapes into the fadd, an error will be raised.
#
#   We can do more specializations. For example, we can write
#   :code:`n = tvm.convert(1024)` instead of :code:`n = tvm.var("n")`,
#   in the computation declaration. The generated function will
#   only take vectors with length 1024.
#

######################################################################
# Save Compiled Module
# --------------------
# Besides runtime compilation, we can save the compiled modules into
# file and load them back later. This is called ahead of time compilation.
#
# The following code first does the following step:
#
# - It saves the compiled host module into an object file.
# - Then it saves the device module into a ptx file.
# - cc.create_shared calls a env compiler(gcc) to create a shared library
#
from tvm.contrib import cc
from tvm.contrib import util

temp = util.tempdir()
fadd.save(temp.relpath("myadd.o"))
if tgt == "cuda":
    fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
print(temp.listdir())

######################################################################
# .. note:: Module Storage Format
#
#   The CPU(host) module is directly saved as a shared library(so).
#   There can be multiple customed format on the device code.
#   In our example, device code is stored in ptx, as well as a meta
#   data json file. They can be loaded and linked seperatedly via import.
#

######################################################################
# Load Compiled Module
# --------------------
# We can load the compiled module from the file system and run the code.
# The following code load the host and device module seperatedly and
# re-link them together. We can verify that the newly loaded function works.
#
fadd1 = tvm.module.load(temp.relpath("myadd.so"))
if tgt == "cuda":
    fadd1_dev = tvm.module.load(temp.relpath("myadd.ptx"))
    fadd1.import_module(fadd1_dev)
fadd1(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Pack Everything into One Library
# --------------------------------
# In the above example, we store the device and host code seperatedly.
# TVM also supports export everything as one shared library.
# Under the hood, we pack the device modules into binary blobs and link
# them together with the host code.
# Currently we support packing of Metal, OpenCL and CUDA modules.
#
fadd.export_library(temp.relpath("myadd_pack.so"))
fadd2 = tvm.module.load(temp.relpath("myadd_pack.so"))
fadd2(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# .. note:: Runtime API and Thread-Safety
#
#   The compiled modules of TVM do not depend on the TVM compiler.
#   Instead, it only depends on a minimum runtime library.
#   TVM runtime library wraps the device drivers and provides
#   thread-safe and device agnostic call into the compiled functions.
#
#   This means you can call the compiled TVM function from any thread,
#   on any GPUs.
#

######################################################################
# Generate OpenCL Code
# --------------------
# TVM provides code generation features into multiple backends,
# we can also generate OpenCL code or LLVM code that runs on CPU backends.
#
# The following codeblocks generate opencl code, creates array on opencl
# device, and verifies the correctness of the code.
#
if tgt == "opencl":
    fadd_cl = tvm.build(s, [A, B, C], "opencl", name="myadd")
    print("------opencl code------")
    print(fadd_cl.imported_modules[0].get_source())
    ctx = tvm.cl(0)
    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
    fadd_cl(a, b, c)
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Summary
# -------
# This tutorial provides a walk through of TVM workflow using
# a vector add example. The general workflow is
#
# - Describe your computation via series of operations.
# - Describe how we want to compute use schedule primitives.
# - Compile to the target function we want.
# - Optionally, save the function to be loaded later.
#
# You are more than welcomed to checkout other examples and
# tutorials to learn more about the supported operations, schedule primitives
# and other features in TVM.
#

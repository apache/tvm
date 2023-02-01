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

Working with Operators Using Tensor Expression
==============================================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

In this tutorial we will turn our attention to how TVM works with Tensor
Expression (TE) to define tensor computations and apply loop optimizations. TE
describes tensor computations in a pure functional language (that is each
expression has no side effects). When viewed in context of the TVM as a whole,
Relay describes a computation as a set of operators, and each of these
operators can be represented as a TE expression where each TE expression takes
input tensors and produces an output tensor.

This is an introductory tutorial to the Tensor Expression language in TVM. TVM
uses a domain specific tensor expression for efficient kernel construction. We
will demonstrate the basic workflow with two examples of using the tensor expression
language. The first example introduces TE and scheduling with vector
addition. The second expands on these concepts with a step-by-step optimization
of a matrix multiplication with TE. This matrix multiplication example will
serve as the comparative basis for future tutorials covering more advanced
features of TVM.
"""


################################################################################
# Example 1: Writing and Scheduling Vector Addition in TE for CPU
# ---------------------------------------------------------------
#
# Let's look at an example in Python in which we will implement a TE for
# vector addition, followed by a schedule targeted towards a CPU.
# We begin by initializing a TVM environment.

import tvm
import tvm.testing
from tvm import te
import numpy as np

################################################################################
# You will get better performance if you can identify the CPU you are targeting
# and specify it. If you're using LLVM, you can get this information from the
# command ``llc --version`` to get the CPU type, and you can check
# ``/proc/cpuinfo`` for additional extensions that your processor might
# support. For example, you can use ``llvm -mcpu=skylake-avx512`` for CPUs with
# AVX-512 instructions.

tgt = tvm.target.Target(target="llvm", host="llvm")

################################################################################
# Describing the Vector Computation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We describe a vector addition computation. TVM adopts tensor semantics, with
# each intermediate result represented as a multi-dimensional array. The user
# needs to describe the computation rule that generates the tensors. We first
# define a symbolic variable ``n`` to represent the shape. We then define two
# placeholder Tensors, ``A`` and ``B``, with given shape ``(n,)``. We then
# describe the result tensor ``C``, with a ``compute`` operation. The
# ``compute`` defines a computation, with the output conforming to the
# specified tensor shape and the computation to be performed at each position
# in the tensor defined by the lambda function. Note that while ``n`` is a
# variable, it defines a consistent shape between the ``A``, ``B`` and ``C``
# tensors. Remember, no actual computation happens during this phase, as we
# are only declaring how the computation should be done.

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

################################################################################
# .. admonition:: Lambda Functions
#
#   The second argument to the ``te.compute`` method is the function that
#   performs the computation. In this example, we're using an anonymous function,
#   also known as a ``lambda`` function, to define the computation, in this case
#   addition on the ``i``\th element of ``A`` and ``B``.

################################################################################
# Create a Default Schedule for the Computation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# While the above lines describe the computation rule, we can compute ``C`` in
# many different ways to fit different devices. For a tensor with multiple
# axes, you can choose which axis to iterate over first, or computations can be
# split across different threads. TVM requires that the user to provide a
# schedule, which is a description of how the computation should be performed.
# Scheduling operations within TE can change loop orders, split computations
# across different threads, and group blocks of data together, amongst other
# operations. An important concept behind schedules is that they only describe
# how the computation is performed, so different schedules for the same TE will
# produce the same result.
#
# TVM allows you to create a naive schedule that will compute ``C`` in by
# iterating in row major order.
#
# .. code-block:: c
#
#   for (int i = 0; i < n; ++i) {
#     C[i] = A[i] + B[i];
#   }

s = te.create_schedule(C.op)

######################################################################
# Compile and Evaluate the Default Schedule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# With the TE expression and a schedule, we can produce runnable code for our
# target language and architecture, in this case LLVM and a CPU. We provide
# TVM with the schedule, a list of the TE expressions that are in the schedule,
# the target and host, and the name of the function we are producing. The result
# of the output is a type-erased function that can be called directly from Python.
#
# In the following line, we use ``tvm.build`` to create a function. The build
# function takes the schedule, the desired signature of the function (including
# the inputs and outputs) as well as target language we want to compile to.

fadd = tvm.build(s, [A, B, C], tgt, name="myadd")

################################################################################
# Let's run the function, and compare the output to the same computation in
# numpy. The compiled TVM function exposes a concise C API that can be invoked
# from any language. We begin by creating a device, which is a device (CPU in this
# example) that TVM can compile the schedule to. In this case the device is an
# LLVM CPU target. We can then initialize the tensors in our device and
# perform the custom addition operation. To verify that the computation is
# correct, we can compare the result of the output of the c tensor to the same
# computation performed by numpy.

dev = tvm.device(tgt.kind.name, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

################################################################################
# To get a comparison of how fast this version is compared to numpy, create a
# helper function to run a profile of the TVM generated code.
import timeit

np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "n = 32768\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(n, 1).astype(dtype)\n"
    "b = numpy.random.rand(n, 1).astype(dtype)\n",
    stmt="answer = a + b",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))


def evaluate_addition(func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    n = 32768
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))

    log.append((optimization, mean_time))


log = [("numpy", np_running_time / np_repeat)]
evaluate_addition(fadd, tgt, "naive", log=log)

################################################################################
# Updating the Schedule to Use Parallelism
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we've illustrated the fundamentals of TE, let's go deeper into what
# schedules do, and how they can be used to optimize tensor expressions for
# different architectures. A schedule is a series of steps that are applied to
# an expression to transform it in a number of different ways. When a schedule
# is applied to an expression in TE, the inputs and outputs remain the same,
# but when compiled the implementation of the expression can change. This
# tensor addition, in the default schedule, is run serially but is easy to
# parallelize across all of the processor threads. We can apply the parallel
# schedule operation to our computation.

s[C].parallel(C.op.axis[0])

################################################################################
# The ``tvm.lower`` command will generate the Intermediate Representation (IR)
# of the TE, with the corresponding schedule. By lowering the expression as we
# apply different schedule operations, we can see the effect of scheduling on
# the ordering of the computation. We use the flag ``simple_mode=True`` to
# return a readable C-style statement.

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# It's now possible for TVM to run these blocks on independent threads. Let's
# compile and run this new schedule with the parallel operation applied:

fadd_parallel = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")
fadd_parallel(a, b, c)

tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

evaluate_addition(fadd_parallel, tgt, "parallel", log=log)

################################################################################
# Updating the Schedule to Use Vectorization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Modern CPUs also have the ability to perform SIMD operations on floating
# point values, and we can apply another schedule to our computation expression
# to take advantage of this. Accomplishing this requires multiple steps: first
# we have to split the schedule into inner and outer loops using the split
# scheduling primitive. The inner loops can use vectorization to use SIMD
# instructions using the vectorize scheduling primitive, then the outer loops
# can be parallelized using the parallel scheduling primitive. Choose the split
# factor to be the number of threads on your CPU.

# Recreate the schedule, since we modified it with the parallel operation in
# the previous example
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)

# This factor should be chosen to match the number of threads appropriate for
# your CPU. This will vary depending on architecture, but a good rule is
# setting this factor to equal the number of available CPU cores.
factor = 4

outer, inner = s[C].split(C.op.axis[0], factor=factor)
s[C].parallel(outer)
s[C].vectorize(inner)

fadd_vector = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")

evaluate_addition(fadd_vector, tgt, "vector", log=log)

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Comparing the Different Schedules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can now compare the different schedules

baseline = log[0][1]
print("%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20)))
for result in log:
    print(
        "%s\t%s\t%s"
        % (result[0].rjust(20), str(result[1]).rjust(20), str(result[1] / baseline).rjust(20))
    )


################################################################################
# .. admonition:: Code Specialization
#
#   As you may have noticed, the declarations of ``A``, ``B`` and ``C`` all
#   take the same shape argument, ``n``. TVM will take advantage of this to
#   pass only a single shape argument to the kernel, as you will find in the
#   printed device code. This is one form of specialization.
#
#   On the host side, TVM will automatically generate check code that checks
#   the constraints in the parameters. So if you pass arrays with different
#   shapes into fadd, an error will be raised.
#
#   We can do more specializations. For example, we can write :code:`n =
#   tvm.runtime.convert(1024)` instead of :code:`n = te.var("n")`, in the
#   computation declaration. The generated function will only take vectors with
#   length 1024.

################################################################################
# We've defined, scheduled, and compiled a vector addition operator, which we
# were then able to execute on the TVM runtime. We can save the operator as a
# library, which we can then load later using the TVM runtime.

################################################################################
# Targeting Vector Addition for GPUs (Optional)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TVM is capable of targeting multiple architectures. In the next example, we
# will target compilation of the vector addition to GPUs.

# If you want to run this code, change ``run_cuda = True``
# Note that by default this example is not run in the docs CI.

run_cuda = False
if run_cuda:
    # Change this target to the correct backend for you gpu. For example: cuda (NVIDIA GPUs),
    # rocm (Radeon GPUS), OpenCL (opencl).
    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")

    # Recreate the schedule
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    print(type(C))

    s = te.create_schedule(C.op)

    bx, tx = s[C].split(C.op.axis[0], factor=64)

    ################################################################################
    # Finally we must bind the iteration axis bx and tx to threads in the GPU
    # compute grid. The naive schedule is not valid for GPUs, and these are
    # specific constructs that allow us to generate code that runs on a GPU.

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
    # function. fadd is the generated host wrapper function, it contains
    # a reference to the generated device function internally.

    fadd = tvm.build(s, [A, B, C], target=tgt_gpu, name="myadd")

    ################################################################################
    # The compiled TVM function exposes a concise C API that can be invoked from
    # any language.
    #
    # We provide a minimal array API in python to aid quick testing and prototyping.
    # The array API is based on the `DLPack <https://github.com/dmlc/dlpack>`_ standard.
    #
    # - We first create a GPU device.
    # - Then tvm.nd.array copies the data to the GPU.
    # - ``fadd`` runs the actual computation
    # - ``numpy()`` copies the GPU array back to the CPU (so we can verify correctness).
    #
    # Note that copying the data to and from the memory on the GPU is a required step.

    dev = tvm.device(tgt_gpu.kind.name, 0)

    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    ################################################################################
    # Inspect the Generated GPU Code
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # You can inspect the generated code in TVM. The result of tvm.build is a TVM
    # Module. fadd is the host module that contains the host wrapper, it also
    # contains a device module for the CUDA (GPU) function.
    #
    # The following code fetches the device module and prints the content code.

    if (
        tgt_gpu.kind.name == "cuda"
        or tgt_gpu.kind.name == "rocm"
        or tgt_gpu.kind.name.startswith("opencl")
    ):
        dev_module = fadd.imported_modules[0]
        print("-----GPU code-----")
        print(dev_module.get_source())
    else:
        print(fadd.get_source())

################################################################################
# Saving and Loading Compiled Modules
# -----------------------------------
# Besides runtime compilation, we can save the compiled modules into a file and
# load them back later.
#
# The following code first performs the following steps:
#
# - It saves the compiled host module into an object file.
# - Then it saves the device module into a ptx file.
# - cc.create_shared calls a compiler (gcc) to create a shared library

from tvm.contrib import cc
from tvm.contrib import utils

temp = utils.tempdir()
fadd.save(temp.relpath("myadd.o"))
if tgt.kind.name == "cuda":
    fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
if tgt.kind.name == "rocm":
    fadd.imported_modules[0].save(temp.relpath("myadd.hsaco"))
if tgt.kind.name.startswith("opencl"):
    fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
print(temp.listdir())

################################################################################
# .. admonition:: Module Storage Format
#
#   The CPU (host) module is directly saved as a shared library (.so). There
#   can be multiple customized formats of the device code. In our example, the
#   device code is stored in ptx, as well as a meta data json file. They can be
#   loaded and linked separately via import.

################################################################################
# Load Compiled Module
# ~~~~~~~~~~~~~~~~~~~~
# We can load the compiled module from the file system and run the code. The
# following code loads the host and device module separately and links them
# together. We can verify that the newly loaded function works.

fadd1 = tvm.runtime.load_module(temp.relpath("myadd.so"))
if tgt.kind.name == "cuda":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.ptx"))
    fadd1.import_module(fadd1_dev)

if tgt.kind.name == "rocm":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.hsaco"))
    fadd1.import_module(fadd1_dev)

if tgt.kind.name.startswith("opencl"):
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.cl"))
    fadd1.import_module(fadd1_dev)

fadd1(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

################################################################################
# Pack Everything into One Library
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In the above example, we store the device and host code separately. TVM also
# supports export everything as one shared library. Under the hood, we pack
# the device modules into binary blobs and link them together with the host
# code. Currently we support packing of Metal, OpenCL and CUDA modules.

fadd.export_library(temp.relpath("myadd_pack.so"))
fadd2 = tvm.runtime.load_module(temp.relpath("myadd_pack.so"))
fadd2(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

################################################################################
# .. admonition:: Runtime API and Thread-Safety
#
#   The compiled modules of TVM do not depend on the TVM compiler. Instead,
#   they only depend on a minimum runtime library. The TVM runtime library
#   wraps the device drivers and provides thread-safe and device agnostic calls
#   into the compiled functions.
#
#   This means that you can call the compiled TVM functions from any thread, on
#   any GPUs, provided that you have compiled the code for that GPU.

################################################################################
# Generate OpenCL Code
# --------------------
# TVM provides code generation features into multiple backends. We can also
# generate OpenCL code or LLVM code that runs on CPU backends.
#
# The following code blocks generate OpenCL code, creates array on an OpenCL
# device, and verifies the correctness of the code.

if tgt.kind.name.startswith("opencl"):
    fadd_cl = tvm.build(s, [A, B, C], tgt, name="myadd")
    print("------opencl code------")
    print(fadd_cl.imported_modules[0].get_source())
    dev = tvm.cl(0)
    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd_cl(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

################################################################################
# .. admonition:: TE Scheduling Primitives
#
#   TVM includes a number of different scheduling primitives:
#
#   - split: splits a specified axis into two axises by the defined factor.
#   - tile: tiles will split a computation across two axes by the defined factors.
#   - fuse: fuses two consecutive axises of one computation.
#   - reorder: can reorder the axises of a computation into a defined order.
#   - bind: can bind a computation to a specific thread, useful in GPU programming.
#   - compute_at: by default, TVM will compute tensors at the outermost level
#     of the function, or the root, by default. compute_at specifies that one
#     tensor should be computed at the first axis of computation for another
#     operator.
#   - compute_inline: when marked inline, a computation will be expanded then
#     inserted into the address where the tensor is required.
#   - compute_root: moves a computation to the outermost layer, or root, of the
#     function. This means that stage of the computation will be fully computed
#     before it moves on to the next stage.
#
#   A complete description of these primitives can be found in the
#   :ref:`Schedule Primitives <schedule_primitives>` docs page.

################################################################################
# Example 2: Manually Optimizing Matrix Multiplication with TE
# ------------------------------------------------------------
#
# Now we will consider a second, more advanced example, demonstrating how with
# just 18 lines of python code TVM speeds up a common matrix multiplication operation by 18x.
#
# **Matrix multiplication is a compute intensive operation. There are
# two important optimizations for good CPU performance:**
#
# 1. Increase the cache hit rate of memory access. Both complex
#    numerical computation and hot-spot memory access can be
#    accelerated by a high cache hit rate. This requires us to
#    transform the origin memory access pattern to a pattern that fits
#    the cache policy.
#
# 2. SIMD (Single instruction multi-data), also known as the vector
#    processing unit. On each cycle instead of processing a single
#    value, SIMD can process a small batch of data.  This requires us
#    to transform the data access pattern in the loop body in uniform
#    pattern so that the LLVM backend can lower it to SIMD.
#
# The techniques used in this tutorial are a subset of tricks mentioned in this
# `repository <https://github.com/flame/how-to-optimize-gemm>`_. Some of them
# have been applied by TVM abstraction automatically, but some of them cannot
# be automatically applied due to TVM constraints.

################################################################################
# Preparation and Performance Baseline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We begin by collecting performance data on the `numpy` implementation of
# matrix multiplication.

import tvm
import tvm.testing
from tvm import te
import numpy

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024

# The default tensor data type in tvm
dtype = "float32"

# You will want to adjust the target to match any CPU vector extensions you
# might have. For example, if you're using using Intel AVX2 (Advanced Vector
# Extensions) ISA for SIMD, you can get the best performance by changing the
# following line to ``llvm -mcpu=core-avx2``, or specific type of CPU you use.
# Recall that you're using llvm, you can get this information from the command
# ``llc --version`` to get the CPU type, and you can check ``/proc/cpuinfo``
# for additional extensions that your processor might support.

target = tvm.target.Target(target="llvm", host="llvm")
dev = tvm.device(target.kind.name, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

# Repeatedly perform a matrix multiplication to get a performance baseline
# for the default numpy implementation
np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))

answer = numpy.dot(a.numpy(), b.numpy())

################################################################################
# Now we write a basic matrix multiplication using TVM TE and verify that it
# produces the same results as the numpy implementation. We also write a
# function that will help us measure the performance of the schedule
# optimizations.

# TVM Matrix Multiplication using TE
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

# Default schedule
s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name="mmult")

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)


def evaluate_operation(s, vars, target, name, optimization, log):
    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))
    log.append((optimization, mean_time))


log = []

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="none", log=log)

################################################################################
# Let's take a look at the intermediate representation of the operator and
# default schedule using the TVM lower function. Note how the implementation is
# essentially a naive implementation of a matrix multiplication, using three
# nested loops over the indices of the A and B matrices.

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 1: Blocking
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# A important trick to enhance the cache hit rate is blocking, where you
# structure memory access such that the inside a block is a small neighborhood
# that has high memory locality. In this tutorial, we pick a block factor of
# 32. This will result in a block that will fill a 32 * 32 * sizeof(float) area
# of memory. This corresponds to a cache size of 4KB, in relation to a
# reference cache size of 32 KB for L1 cache.
#
# We begin by creating a default schedule for the ``C`` operation, then apply a
# ``tile`` scheduling primitive to it with the specified block factor, with the
# scheduling primitive returning the resulting loop order from outermost to
# innermost, as a vector ``[x_outer, y_outer, x_inner, y_inner]``. We then get
# the reduction axis for output of the operation, and perform a split operation
# on it using a factor of 4. This factor doesn't directly impact the blocking
# optimization we're working on right now, but will be useful later when we
# apply vectorization.
#
# Now that the operation has been blocked, we can reorder the computation to
# put the reduction operation into the outermost loop of the computation,
# helping to guarantee that the blocked data remains in cache. This completes
# the schedule, and we can build and test the performance compared to the naive
# schedule.

bn = 32

# Blocking by loop tiling
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# Hoist reduction domain outside the blocking loop
s[C].reorder(xo, yo, ko, ki, xi, yi)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="blocking", log=log)

################################################################################
# By reordering the computation to take advantage of caching, you should see a
# significant improvement in the performance of the computation. Now, print the
# internal representation and compare it to the original:

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 2: Vectorization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Another important optimization trick is vectorization. When the memory access
# pattern is uniform, the compiler can detect this pattern and pass the
# continuous memory to the SIMD vector processor. In TVM, we can use the
# ``vectorize`` interface to hint the compiler this pattern, taking advantage
# of this hardware feature.
#
# In this tutorial, we chose to vectorize the inner loop row data since it is
# already cache friendly from our previous optimizations.

# Apply the vectorization optimization
s[C].vectorize(yi)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="vectorization", log=log)

# The generalized IR after vectorization
print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 3: Loop Permutation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# If we look at the above IR, we can see the inner loop row data is vectorized
# and B is transformed into PackedB (this is evident by the `(float32x32*)B2`
# portion of the inner loop). The traversal of PackedB is sequential now. So we
# will look at the access pattern of A. In current schedule, A is accessed
# column by column which is not cache friendly. If we change the nested loop
# order of `ki` and inner axes `xi`, the access pattern for A matrix will be
# more cache friendly.

s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# re-ordering
s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

evaluate_operation(
    s, [A, B, C], target=target, name="mmult", optimization="loop permutation", log=log
)

# Again, print the new generalized IR
print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 4: Array Packing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Another important trick is array packing. This trick is to reorder the
# storage dimension of the array to convert the continuous access pattern on
# certain dimension to a sequential pattern after flattening.
#
# .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/array-packing.png
#    :align: center
#
# Just as it is shown in the figure above, after blocking the computations, we
# can observe the array access pattern of B (after flattening), which is
# regular but discontinuous. We expect that after some transformation we can
# get a continuous access pattern. By reordering a ``[16][16]`` array to a
# ``[16/4][16][4]`` array the access pattern of B will be sequential when
# grabbing the corresponding value from the packed array.
#
# To accomplish this, we are going to have to start with a new default
# schedule, taking into account the new packing of B. It's worth taking a
# moment to comment on this: TE is a powerful and expressive language for
# writing optimized operators, but it often requires some knowledge of the
# underlying algorithm, data structures, and hardware target that you are
# writing for. Later in the tutorial, we will discuss some of the options for
# letting TVM take that burden. Regardless, let's move on with the new
# optimized schedule.

# We have to re-write the algorithm slightly.
packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
C = te.compute(
    (M, N),
    lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
    name="C",
)

s = te.create_schedule(C.op)

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="array packing", log=log)

# Here is the generated IR after array packing.
print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 5: Optimizing Block Writing Through Caching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Up to this point all of our optimizations have focused on efficiently
# accessing and computing the data from the `A` and `B` matrices to compute the
# `C` matrix. After the blocking optimization, the operator will write result
# to `C` block by block, and the access pattern is not sequential. We can
# address this by using a sequential cache array, using a combination of
# `cache_write`, `compute_at`, and `unroll`to hold the block results and write
# to `C` when all the block results are ready.

s = te.create_schedule(C.op)

# Allocate write cache
CC = s.cache_write(C, "global")

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

# Write cache is computed at yo
s[CC].compute_at(s[C], yo)

# New inner axes
xc, yc = s[CC].op.axis

(k,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(k, factor=4)
s[CC].reorder(ko, xc, ki, yc)
s[CC].unroll(ki)
s[CC].vectorize(yc)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="block caching", log=log)

# Here is the generated IR after write cache blocking.
print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 6: Parallelization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# So far, our computation is only designed to use a single core. Nearly all
# modern processors have multiple cores, and computation can benefit from
# running computations in parallel. The final optimization is to take advantage
# of thread-level parallelization.

# parallel
s[C].parallel(xo)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(
    s, [A, B, C], target=target, name="mmult", optimization="parallelization", log=log
)

# Here is the generated IR after parallelization.
print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Summary of Matrix Multiplication Example
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# After applying the above simple optimizations with only 18 lines of code, our
# generated code can begin to approach the performance of `numpy` with the Math
# Kernel Library (MKL). Since we've been logging the performance as we've been
# working, we can compare the results.

baseline = log[0][1]
print("%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20)))
for result in log:
    print(
        "%s\t%s\t%s"
        % (result[0].rjust(20), str(result[1]).rjust(20), str(result[1] / baseline).rjust(20))
    )

################################################################################
# Note that the outputs on the web page reflect the running times on a
# non-exclusive Docker container, and should be considered unreliable. It is
# highly encouraged to run the tutorial by yourself to observe the performance
# gain achieved by TVM, and to carefully work through each example to
# understand the iterative improvements that are made to the matrix
# multiplication operation.

################################################################################
# Final Notes and Summary
# -----------------------
# As mentioned earlier, how to apply optimizations using TE and scheduling
# primitives can require some knowledge of the underlying architecture and
# algorithms. However, TE was designed to act as a foundation for more complex
# algorithms that can search the potential optimization. With the knowledge you
# have from this introduction to TE, we can now begin to explore how TVM can
# automate the schedule optimization process.
#
# This tutorial provided a walk-through of TVM Tensor Expression (TE) workflow
# using a vector add and a matrix multiplication examples. The general workflow
# is
#
# - Describe your computation via a series of operations.
# - Describe how we want to compute use schedule primitives.
# - Compile to the target function we want.
# - Optionally, save the function to be loaded later.
#
# Upcoming tutorials expand on the matrix multiplication example, and show how
# you can build generic templates of the matrix multiplication and other
# operations with tunable parameters that allows you to automatically optimize
# the computation for specific platforms.

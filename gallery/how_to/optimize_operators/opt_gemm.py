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
.. _opt-gemm:

How to optimize GEMM on CPU
===========================
**Author**: `Jian Weng <https://github.com/were>`_, \
            `Ruofei Yu <https://github.com/yuruofeifei>`_

(TL;DR) TVM provides abstract interfaces which allows users to depict an algorithm and the
algorithm's implementing organization (the so-called schedule) separately. Typically, writing
algorithm in high-performance schedule breaks the algorithm's readability and modularity. Also,
trying various seemingly promising schedules is time-consuming. With the help of TVM, we can
try these schedules efficiently to enhance the performance.

In this tutorial, we will demonstrate how to use TVM to optimize square matrix multiplication
and achieve 200 times faster than baseline by simply adding 18 extra lines of code.

There are two important optimizations on intense computation applications executed on CPU:
    1. Increase the cache hit rate of memory access. Both complex numerical computation and hot-spot
       memory access can be accelerated from high cache hit rate. This requires us to transform the
       origin memory access pattern to the pattern fits the cache policy.
    2. SIMD (Single instruction multi-data), or we call it vector processing unit. Every time, a
       small batch of data, rather than a single grid, will be processed. This requires us to
       transform the data access pattern in the loop body in uniform pattern so that the LLVM
       backend can lower it to SIMD.

Actually, all the methodologies used in this tutorial is a subset of tricks mentioned in this
`repo <https://github.com/flame/how-to-optimize-gemm>`_. Some of them have been applied by TVM
abstraction automatically, but some of them cannot be simply applied due to TVM constraints.

All the experiment results mentioned below, are executed on 2015's 15' MacBook equipped with
Intel i7-4770HQ CPU. The cache line size should be 64 bytes for all the x86 CPUs.
"""


################################################################################################
# Preparation and Baseline
# ------------------------
# In this tutorial, we will demo how to use TVM to optimize matrix multiplication.
# Before actually demonstrating, we first define these variables.
# Then we write a baseline implementation, the simplest way to write a matrix multiplication in TVM.

import tvm
import tvm.testing
from tvm import te
import numpy
import timeit

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024

# The default tensor type in tvm
dtype = "float32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = "llvm"
dev = tvm.device(target, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

np_repeat = 100
np_runing_time = timeit.timeit(
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
print("Numpy running time: %f" % (np_runing_time / np_repeat))

answer = numpy.dot(a.numpy(), b.numpy())

# Algorithm
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

# Default schedule
s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("Baseline: %f" % evaluator(a, b, c).mean)

################################################################################################
# In TVM, we can always inspect lower level IR to debug or optimize our schedule.
# Here is the generated IR using our baseline schedule.

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################################
# Blocking
# --------
# A important trick to enhance the cache hit rate is blocking --- data chunk will be computed
# block by block. The memory access inside the block is a small neighbourhood which is with high
# memory locality. In this tutorial, I picked up 32 as the blocking factor. So the block will
# fill 32 * 32 * sizeof(float) which is 4KB in the cache whose total size is 32KB (L1 data cache)

bn = 32
kfactor = 4
s = te.create_schedule(C.op)

# Blocking by loop tiling
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

# Hoist reduction domain outside the blocking loop
s[C].reorder(mo, no, ko, ki, mi, ni)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

# By simply tiling the loop 32x32, and hoisting ko, ki outside the blocking loops,
# we can see big speedup compared with the baseline.
evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt1: %f" % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after blocking.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Vectorization
# -------------
# Another important trick is vectorization. When the memory access pattern is uniform,
# the compiler can detect this pattern and pass the continuous memory to vector processor. In TVM,
# we can use `vectorize` interface to hint the compiler this pattern, so that we can accelerate it
# vastly.
#
# In this tutorial, we chose to vectorize the inner loop row data since it is cache friendly.

s = te.create_schedule(C.op)
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

s[C].reorder(mo, no, ko, ki, mi, ni)

# Vectorization
s[C].vectorize(ni)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt2: %f" % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after vectorization.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Loop Permutation
# ----------------
# If we look at the above IR, we can see the inner loop row data is vectorized for both B and C.
# Next we will look at the access pattern of A. In current schedule, A is accessed column by column
# which is not cache friendly. If we change the nested loop order of ki and inner axes mi,
# the access pattern for A matrix is more cache friendly.

s = te.create_schedule(C.op)
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

# re-ordering
s[C].reorder(mo, no, ko, mi, ki, ni)
s[C].vectorize(ni)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt3: %f" % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after loop permutation.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Array Packing
# -------------
# Another important trick is array packing. The trick is to reorder the storage of a multi-
# dimensional array so that it is accessed sequentially after it is flattened and stored in one-
# dimensional memory.
#
# .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/array-packing.png
#      :align: center
#
# NOTE: This figure is a general illustration of how array packing works.


###################################################################################################
# We can use array packing to address the access pattern for B. Observe the array access pattern of
# B after flattening which is not sequential as we iterate over the K dimension. We can reorder B
# with dimensions [K][N] so that it has dimensions [N/bn][K][bn] where bn is the blocking factor and
# also the vector size for B in the inner loop.  This reorder splits N into two dimensions ---
# bigN (N/bn) and littleN (bn) --- and the new dimensions [N/bn][K][bn] match the indexing of B
# from outer to inner loops (no, ko, ki, ni) resulting in a sequential access pattern for B after
# flattening.


# We have to re-write the algorithm slightly.
packedB = te.compute(
    (N / bn, K, bn), lambda bigN, k, littleN: B[k, bigN * bn + littleN], name="packedB"
)
C = te.compute(
    (M, N),
    lambda m, n: te.sum(A[m, k] * packedB[n // bn, k, tvm.tir.indexmod(n, bn)], axis=k),
    name="C",
)

s = te.create_schedule(C.op)

mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

s[C].reorder(mo, no, ko, mi, ki, ni)
s[C].vectorize(ni)

bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt4: %f" % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after array packing.

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################################
# Write cache for blocks
# ----------------------
# After blocking, the program will write result to C block by block, the access pattern
# is not sequential. So we can use a sequential cache array to hold the block results and
# write to C when all the block results are ready.
#

s = te.create_schedule(C.op)

# Allocate write cache
CC = s.cache_write(C, "global")

mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

# Write cache is computed at no
s[CC].compute_at(s[C], no)

# New inner axes
mc, nc = s[CC].op.axis

(kaxis,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(kaxis, factor=kfactor)
s[CC].reorder(ko, mc, ki, nc)
s[CC].vectorize(nc)

# TODO: Add separate optimization step to discuss loop unrolling
# unrolling is a loop optimization strategy which can reduce branch
# prediction failures and increases the chance of concurrent execution
# unroll kfactor loops
s[CC].unroll(ki)

bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt5: %f" % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after blocking.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Parallel
# --------
# Furthermore, we can also utilize multi-core processors to do the thread-level parallelization.

s = te.create_schedule(C.op)

CC = s.cache_write(C, "global")

mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

s[CC].compute_at(s[C], no)

mc, nc = s[CC].op.axis

(kaxis,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(kaxis, factor=kfactor)
s[CC].reorder(ko, mc, ki, nc)
s[CC].vectorize(nc)
s[CC].unroll(ki)

# parallel
s[C].parallel(mo)

bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=50)
opt6_time = evaluator(a, b, c).mean
print("Opt6: %f" % opt6_time)

################################################################################################
# Here is the generated IR after parallelization.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################

##################################################################################################
# Summary
# -------
# After applying the above simple optimizations with only 18 lines of code,
# our generated code can achieve 60% of the `numpy` performance with MKL.
# Note that the outputs on the web page reflect the running times on a non-exclusive
# Docker container, thereby they are *unreliable*. It is highly encouraged to run the
# tutorial by yourself to observe the performance gain achieved by TVM.

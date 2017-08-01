"""
How to optimize GEMM on CPU
===========================
**Author**: `Jian Weng <https://github.com/were>`_

(TL;DR) TVM provides abstract interfaces which allows users to depict an algorithm and the
algorithm's implementing organization (the so-called schedule) separately. Typically, writing
algorithm in high-performance schedule breaks the algorithm's readability and modularity. Also,
trying various seemingly promising schedules is time-consuming. With the help of TVM, we can
try these schedules efficiently to enhance the performance.

In this tutorial, we will demonstrate how square matrix multiplication is optimized step by step by
writing TVM.

There are two important optmizations on intense computation applications executed on CPU:
    1. Increase the cache hit rate of memory access. Both complex numerical computation and hot-spot
       memory access can be acclerated from high cache hit rate. This requires us to transform the
       origin memory access pattern to the pattern fits the cache policy.
    2. SIMD (Single instruction multi-data), or we call it vector processing unit. Every time, a
       small batch of data, rather than a single grid, will be processed. This requires us to
       transform the data access pattern in the loop body in uniform pattern so that the LLVM
       backend can lower it to SIMD.

Actually, all the methodologies used in this tutorial is a subset of tricks mentioned in this
`repo <https://github.com/flame/how-to-optimize-gemm>`_. Some of them have been applied by TVM
abstraction automatically, but some of them cannot be simply applied due to TVM constraints.

All the experiment results mentioned below, are executed on 2013's 15' MacBook equiped with
Intel i7-2760QM CPU. The cache line size should be 64 bytes for all the x86 CPU.
"""

###############################################################################
# Preparation and Baseline
# ------------------------
# In this tutorial we assume all the matrix tensors are square and fix-bounded.
# We use 1024x1024 float32 matrix in demonstration. Before actually demonstrating,
# we first define these variables. Then we write a baseline implementation,
# the simplest way to write a matrix mulplication in TVM.
#

import tvm
import numpy
import time

# The size of the square matrix
N = 1024
# The default tensor type in tvm
dtype = "float32"
# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
b = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
# The expected answer
answer = numpy.dot(a.asnumpy(), b.asnumpy())

# Algorithm
k = tvm.reduce_axis((0, N), 'k')
A = tvm.placeholder((N, N), name = 'A')
B = tvm.placeholder((N, N), name = 'B')
C = tvm.compute(
           A.shape,
           lambda x, y: tvm.sum(A[x, k] * B[k, y], axis = k),
           name = 'C')

# Default schedule
s = tvm.create_schedule(C.op)
func = tvm.build(s, [A, B, C], name = 'mmult')
assert func
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 1)
c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
print('Baseline: %f' % evaluator(a, b, c).mean)

################################################################################################
# Blocking
# --------
# A important trick to enhance the cache hit rate is blocking --- data chunck will be computed
# block by block. The memory access inside the block is a small neighbourhood which is with high
# meomry locality. In this tutorial, I pick up 8, a relatively small value (8 ints < 64 bytes),
# as the blocking size.
#

bn = 8
# Blocking by loop tiling
yo, xo, yi, xi = s[C].tile(C.op.axis[1], C.op.axis[0], bn, bn)
# Hoist reduction domain outside the blocking loop
s[C].reorder(yo, xo, k, yi, xi)
func = tvm.build(s, [A, B, C], name = 'mmult')
assert func
# By simply tiling the loop 8x8, and hoisting k outside the blocking loops, we can get nearly 4x
# speedup compared with the baseline.
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
print('Opt1: %f' % evaluator(a, b, c).mean)

###################################################################################################
# Vectorization
# -------------
# Another important trick is vectorization. When the memory access pattern is uniform, the compiler
# can detect this pattern and pass the continuous memory to vector processor. In TVM, we can use
# `vectorize` interface to hint the compiler this pattern, so that we can accelerate it vastly.
#

# After trying different schedule, we finally found that we can benefit from vectorizing
# the row loop most, i.e. yi.
s[C].vectorize(yi)
func = tvm.build(s, [A, B, C], name = 'mmult')
assert func
# We can get almost another 4x speedup compared with the previous schedule.
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
print('Opt2: %f' % evaluator(a, b, c).mean)

###################################################################################################
# Array Packing
# -------------
# Another important trick is array packing. This trick is to reorder the storage dimension of the
# array to convert the continuous access pattern on certain dimension to a sequential pattern after
# flattening.
#
# .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/array-packing.png
#      :align: center
#      :scale: 100%
#


###################################################################################################
# Just as it is shown in the figure above, after blocking the computations, we can observe the array
# access pattern of B (after flattening), which is regular but discontinuous. We expect that after
# some transformation we can get continuous access pattern. We can reorder a [16][16] array to 
# a [16/4][16][4] array, so that the access pattern of B will be sequential when grabing 
# the corresponding value from the packed array.
#

# We have to re-write the algorithm slightly.
packedB = tvm.compute((N / bn, N, bn), lambda x, y, z: B[y, x * bn + z], name = 'packedB')
C = tvm.compute(A.shape,
                lambda x, y: tvm.sum(A[x, k] * packedB[y / bn, k, y % bn], axis = k),
                name = 'C')

# Same schedule
s = tvm.create_schedule(C.op)
yo, xo, yi, xi = s[C].tile(C.op.axis[1], C.op.axis[0], bn, bn)
s[C].reorder(yo, xo, k, yi, xi)
s[C].vectorize(yi)

func = tvm.build(s, [A, B, C], name = 'mmult')
assert func
# We can accelerate it almost 3x compared with the previous schedule.
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
print('Opt3: %f' % evaluator(a, b, c).mean)

##################################################################################################
# Summary
# -------
# After applying three main tricks, we can achieve almost 90% performance of numpy.
# Further observation is required to catch up with the performance of numpy.
#

# TODO(Jian Weng): Catch up with the performance of numpy.
_a = a.asnumpy()
_b = b.asnumpy()
now = time.clock()
answer = numpy.dot(_a, _b)
print("Numpy: %f" % (time.clock() - now))


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

All the experiment results mentioned below, are executed on 2015's 15' MacBook equiped with
Intel i7-4770QH CPU. The cache line size should be 64 bytes for all the x86 CPU.
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

now = time.time()
# The expected answer
answer = numpy.dot(a.asnumpy(), b.asnumpy())
np_runing_time = time.time() - now
print("Numpy running time: %f" % np_runing_time)

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

c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number=1)
print('Baseline: %f' % evaluator(a, b, c).mean)

################################################################################################
# In TVM, we can always inspect lower level IR to debug or optimize our schedule.
# Here is the generated IR using our baseline schedule.

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################################
# Blocking
# --------
# A important trick to enhance the cache hit rate is blocking --- data chunck will be computed
# block by block. The memory access inside the block is a small neighbourhood which is with high
# memory locality. In this tutorial, I picked up 32 as the blocking factor. So the block will
# fill 32 * 32 * sizeof(int) bytes in the cache whose total size is 32KB (L1 data cache)

bn = 32
s = tvm.create_schedule(C.op)
# Blocking by loop tiling
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
# Hoist reduction domain outside the blocking loop
s[C].reorder(xo, yo, k, xi, yi)
func = tvm.build(s, [A, B, C], name = 'mmult')
assert func

c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

# By simply tiling the loop 32x32, and hoisting k outside the blocking loops, we can see big
# speedup compared with the baseline.
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number=5)
print('Opt1: %f' % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after blocking.

print(tvm.lower(s, [A, B, C], simple_mode=True))

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

s = tvm.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
s[C].reorder(xo, yo, k, xi, yi)

func = tvm.build(s, [A, B, C], name = 'mmult')
assert func

c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number=5)
print('Opt2: %f' % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after array packing.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Vectorization
# -------------
# Another important trick is vectorization. When the memory access pattern is uniform,
# the compiler can detect this pattern and pass the continuous memory to vector processor. In TVM,
# we can use `vectorize` interface to hint the compiler this pattern, so that we can accelerate it vastly.
#
# In this tutorial, we chose to vectorize the inner loop row data since it is cache friendly.

s = tvm.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
s[C].reorder(xo, yo, k, xi, yi)

# Vectorization
s[C].vectorize(yi)
func = tvm.build(s, [A, B, C], name = 'mmult')
assert func

c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number=5)
print('Opt3: %f' % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after vectorization.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Loop Permutation
# -------------
# If we look at the above IR, we can see the inner loop row data is vectorized and
# B is transformed into PackedB. The traversal of PackedB is sequential now.
# So we will look at the access pattern of A. In current schedule, A is accessed column by column
# which is not cache friendly. If we change the nested loop order of k and inner row index,
# the access pattern for A matrix is more cache friendly.

s = tvm.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
s[C].reorder(xo, yo, xi, k, yi)

# Vectorization
s[C].vectorize(yi)

func = tvm.build(s, [A, B, C], name = 'mmult')
assert func

c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number=5)
print('Opt4: %f' % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after loop permutation.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Parallel
# -------------
# Futhermore, we can also utilize multi-core processors to parallelize computation.

s = tvm.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
s[C].reorder(xo, yo, xi, k, yi)
s[C].vectorize(yi)

# parallel
s[C].parallel(xo)

func = tvm.build(s, [A, B, C], name = 'mmult')
assert func

c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
opt5_time = evaluator(a, b, c).mean
print('Opt5: %f' % opt5_time)

##################################################################################################
# Summary
# -------
# After applying the above simple optimizations with only 9 lines of code,
# our generated code can achieve 30% of numpy performance with Apple implemented BLAS.
#
# We can see TVM is very powerful tool to optimize low level computation.


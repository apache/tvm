"""
Optimize GEMM
=============
**Author**: `Jian Weng <https://github.com/were>_`

(TL;DR) TVM provides abstract interfaces which allows users to depict an algorithm and the
algorithm's implementing organization (the so-called schedule) separately. Typically, writing
algorithm in high-performance schedule breaks the algorithm's readability and modularity. Also,
trying various seemingly promising schedules is time-consuming. With the help of TVM, we can
try these schedules efficiently to enhance the performance.

In this tutorial, we will demonstrate how squre matrix multiplication is optimized step by step by
writing TVM.

There are two important optmizations on intense computation applications executed on CPU:
    1. Increase the cache hit rate of memory access. Both complex numerical computation and hot-spot
       memory access can be acclerated from high cache hit rate. This requires us to transform the
       origin memory access pattern to the pattern fits the cache policy.
    2. SIMD (Single instruction multi-data), or we call it vector processing unit. Every time, a
       small batch of data, rather than a single grid, will be processed. This requires us to
       transform the data access pattern in the loop body in uniform pattern so that the LLVM
       backend can lower it to SIMD.

Actually, all the methodologies used in this tutorial is a subset of tricks mentioned in this [repo]
(https://github.com/flame/how-to-optimize-gemm).

All the experiment results mentioned below, are executed on 2013's 15' MacBook equiped
Intel i7-2760QM CPU. The cache line size should be 64 bytes for all the x86 CPU.
"""

import tvm
import numpy
import time

# The size of the squre matrix
N = 1024
# The default tensor type in tvm
dtype = "float32"
# Random generated tensor for testing
a =  tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
b = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
# The expected answer
answer = numpy.dot(a.asnumpy(), b.asnumpy())

# (TL;DR) Before actually discussing about those acceleration tricks, we first define a timer
# function and random generate two square matrix so that we can observe the enhancement easily 
# in later discussion.
def timer(mmult, cnt = 10):
    global N, dtype, a, b, answer
    res = 0
    c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
    #When first-time executing the module, it will take extra time to allocate memory.
    for i in xrange(cnt + 1):
        now = time.clock()
        mmult(a, b, c)
        if i:
            res += time.clock() - now
    numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
    print str(res / float(cnt)) + 's executed!'

# This is the baseline, the simplest way to write a matrix mulplication in TVM.
class GEMM(object):
    def __init__(self, N):
        self.k = tvm.reduce_axis((0, N), 'k')
        self.A = tvm.placeholder((N, N), name = 'A')
        self.B = tvm.placeholder((N, N), name = 'B')
        self.C = tvm.compute(
                    self.A.shape,
                    lambda x, y: tvm.sum(self.A[x, self.k] * self.B[self.k, y], axis = self.k),
                    name = 'C')

    def build(self, show_ir = False):
        # The default schedule
        s = tvm.create_schedule(self.C.op)
        return tvm.build(s, [self.A, self.B, self.C], name = 'mmult')

baseline = GEMM(N)
func = baseline.build()
assert func
# It is too time consuming to run the baseline 10 times, so we just run it once!
print 'Baseline:',
timer(func, 1)

# A important trick to enhance the cache hit rate is blocking --- data chunck will be computed
# block by block. The memory access inside the block is a small neighbourhood which is with high
# meomry locality. In this tutorial, I pick up 8, a relatively small value (8 ints < 64 bytes),
# as the blocking size.
bn = 8

class GEMMopt1(GEMM):
    def __init__(self, N, bn):
        super(GEMMopt1, self).__init__(N)
        self.bn = bn

    def build(self, show_ir = False):
        s = tvm.create_schedule(self.C.op)
        # Blocking by loop tiling
        yo, xo, yi, xi = s[self.C].tile(self.C.op.axis[1], self.C.op.axis[0], self.bn, self.bn)
        # Hoist reduction domain outside the blocking loop
        s[self.C].reorder(yo, xo, self.k, yi, xi)
        return tvm.build(s, [self.A, self.B, self.C], name = 'mmult')

baseline = GEMMopt1(N, bn)
func = baseline.build()
assert func
# By simply tiling the loop 8x8, and hoisting k outside the blocking loops, we can get nearly 4x
# speedup compared with the baseline.
print 'Opt1:',
timer(func, 5)

# Another important trick is vectorization. When the memory access pattern is uniform, the compiler
# can dectect this pattern and pass the continuous memory to vector processor. In TVM, we can use
# vectorize interface to tell the compiler this pattern, so that we can accelerate it vastly.
class GEMMopt2(GEMM):
    def __init__(self, N, bn):
        super(GEMMopt2, self).__init__(N)
        self.bn = bn

    def build(self, show_ir = False):
        s = tvm.create_schedule(self.C.op)
        yo, xo, yi, xi = s[self.C].tile(self.C.op.axis[1], self.C.op.axis[0], self.bn, self.bn)
        s[self.C].reorder(yo, xo, self.k, yi, xi)
        # Actually, in the tutorial 'optimize gemm step by step', they used matrixization --- every
        # time a flattened matrix is passed to vector processor. But in TVM, right now, it is
        # hard to benefit from matrixization. This is mainly caused by some compiler's internal
        # constraints which disables some flexibility in organizing the algorithm.

        # After trying different schedule, we finally found that we can benefit from vectorizing 
        # the row loop most, i.e. yi.
        s[self.C].vectorize(yi)
        return tvm.build(s, [self.A, self.B, self.C], name = 'mmult')

baseline = GEMMopt2(N, bn)
func = baseline.build()
assert func
# We can get almost another 4x speedup compared with the previous schedule.
print 'Opt2:',
timer(func)

# Another important trick is array packing. This trick is to reorder the storage dimension of the
# array to convert the continuous access pattern on certain dimension to a sequential pattern after
# flattening. For the convienience of drawing a figure, we use 4x4 blocking as an example to
# demonstrate array packing:
# First we observe memory access pattern of AB=C:
# A:                   B:                          C:
# ---- ---- ---- ----    |||| **** **** **** ****    ++++ **** **** **** ****
# ---- ---- ---- ----    |||| **** **** **** ****    ++++ **** **** **** ****
# ---- ---- ---- ----    |||| **** **** **** ****    ++++ **** **** **** ****
# ---- ---- ---- ----    |||| **** **** **** ****    ++++ **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# **** **** **** ****    |||| **** **** **** ****    **** **** **** **** ****
# We access A sequentially, but for B, we access it continuous on dimension of rows. Thus, what we 
# want to do is to put this dimension to the inner most dimension. For 1x1 blocking, it is simply
# to transpose the matrix B. However, here is 4x4 case, array B is packed in this fashion:
# B:
#   0123 4567 89AB CDEF        0:  1234  1: 1234  2: 1234  3: 1234
# 0 |||| **** **** ****          0 ||||     ****     ****     ****
# 1 |||| **** **** ****          1 ||||     ****     ****     ****
# 2 |||| **** **** ****          2 ||||     ****     ****     ****
# 3 |||| **** **** ****          3 ||||     ****     ****     ****
# 4 |||| **** **** ****          4 ||||     ****     ****     ****
# 5 |||| **** **** ****          5 ||||     ****     ****     ****
# 6 |||| **** **** ****          6 ||||     ****     ****     ****
# 7 |||| **** **** ****  ->      7 ||||     ****     ****     ****
# 8 |||| **** **** ****          8 ||||     ****     ****     ****
# 9 |||| **** **** ****          9 ||||     ****     ****     ****
# A |||| **** **** ****          A ||||     ****     ****     ****
# B |||| **** **** ****          B ||||     ****     ****     ****
# C |||| **** **** ****          C ||||     ****     ****     ****
# D |||| **** **** ****          D ||||     ****     ****     ****
# E |||| **** **** ****          E ||||     ****     ****     ****
# F |||| **** **** ****          F ||||     ****     ****     ****
# We reorder a 16x16 array to a [16/4][16][4] array so that the access pattern of B will be
# sequential when grabing the corresponding value from the packed array.

class GEMMopt3(object):
    def __init__(self, N, bn):
        # We need to rewrite the algorithm to implement array packing. Some careful observation is
        # required to pack the array and grab the correct corresponding value from the packed array.
        k = tvm.reduce_axis((0, N), 'k')
        A = tvm.placeholder((N, N), name = 'A')
        B = tvm.placeholder((N, N), name = 'B')
        packedB = tvm.compute((N / bn, N, bn), lambda x, y, z: B[y, x * bn + z], name = 'packedB')
        C = tvm.compute(A.shape,
                        lambda x, y: tvm.sum(A[x, k] * packedB[y / bn, k, y % bn], axis = k),
                        name = 'C')

        s = tvm.create_schedule(C.op)
        yo, xo, yi, xi = s[C].tile(C.op.axis[1], C.op.axis[0], bn, bn)
        s[C].reorder(yo, xo, k, yi, xi)
        s[C].vectorize(yi)

        self.func = tvm.build(s, [A, B, C], name = 'mmult')

    def build(self, show_ir = False):
        if show_ir:
            print tvm.lower(s, [A, B, C], simple_mode=True)
        assert self.func
        return self.func

baseline = GEMMopt3(N, bn)
func = baseline.build()
assert func
# We can accelerate it almost 3x compared with the previous schedule.
print 'Opt3:',
timer(func, 10)

# However, we can still get 90% performance compared with numpy.
# TODO(Jian Weng): Catch up with the performance of numpy. Further observation is required.
now = time.clock()
answer = numpy.dot(a.asnumpy(), b.asnumpy())
print "Numpy:", time.clock() - now


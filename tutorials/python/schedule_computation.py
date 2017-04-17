"""
Schedule the Computation with TVM
====================
**Author**: `Ziheng Jiang <https://github.com/ZihengJiang>`_

TVM is a domain specifric language for efficient kernel construction.

In this tutorial, we will show how to schedule the computation in TVM.
"""
from __future__ import absolute_import, print_function

import tvm
import numpy as np

######################################################################
#
# There often exist several methods to compute the same result,
# however, different methods result in diffrent locality and
# performance. So TVM asks user to provide how to execute the
# computation called **Schedule**.
#
# A **Schedule** is a set of transformation of computation that
# transforms the loop of computations in the program.

# declare some variables for use later
n = tvm.var('n')
m = tvm.var('m')

# define a function to print the schedule out
# let's skip the detail for now
def PrintSchedule(s):
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    print(stmt)

######################################################################
# A schedule can be created from a list of ops, by default the
# schedule compute tensor in a serial manner in a row-major order.

# declare a matrix elementwise mul
A = tvm.placeholder((m, n), name='A')
B = tvm.placeholder((m, n), name='B')
T = tvm.compute((m, n), lambda i, j: A[i, j] * B[i, j], name='T')

s = tvm.create_schedule([T.op])
PrintSchedule(s)

######################################################################
# One schedule is composed by multiple stages, and one
# **Stage** represents schedule for one operation. We provide various
# methods to schedule every stage.

######################################################################
# :code:`split` can split a specified axis into two axises by `factor`.
A = tvm.placeholder((m, n), name='A')
T = tvm.compute((m, n), lambda i, j: A[i, j]*2, name='T')

s = tvm.create_schedule(T.op)
outer, inner = s[T].split(T.op.axis[1], factor=32)
PrintSchedule(s)

######################################################################
# :code:`tile` can help you execute the computation tile by tile over
# two axises.
A = tvm.placeholder((m, n), name='A')
T = tvm.compute((m, n), lambda i, j: A[i, j], name='T')

s = tvm.create_schedule(T.op)
xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
PrintSchedule(s)

######################################################################
# :code:`fuse` can fuse two consecutive axises of one computation.
A = tvm.placeholder((m, n), name='A')
T = tvm.compute((m, n), lambda i, j: A[i, j], name='T')

s = tvm.create_schedule(T.op)
# tile to four axises first: (i.outer, j.outer, i.inner, j.inner)
xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
# then fuse (i.inner, j.inner) into one axis: (i.inner.j.inner.fused)
fused = s[T].fuse(yi, xi)
PrintSchedule(s)

######################################################################
# :code:`reorder` can reorder the axises in the specified order.
A = tvm.placeholder((m, n), name='A')
T = tvm.compute((m, n), lambda i, j: A[i, j], name='T')

s = tvm.create_schedule(T.op)
# tile to four axises first: (i.outer, j.outer, i.inner, j.inner)
xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
# then reorder the axises: (i.inner, j.outer, i.outer, j.inner)
s[T].reorder(xi, yo, xo, yi)
PrintSchedule(s)

######################################################################
# :code:`bind` can bind a specified axis with a thread axis, often used
# in gpu programming.
A = tvm.placeholder((n,), name='A')
T = tvm.compute(A.shape, lambda i: A[i] * 2, name='T')

s = tvm.create_schedule(T.op)
bx, tx = s[T].split(T.op.axis[0], factor=64)
s[T].bind(bx, tvm.thread_axis("blockIdx.x"))
s[T].bind(tx, tvm.thread_axis("threadIdx.x"))
PrintSchedule(s)

######################################################################
# For a schedule consists of multiple operators, tvm will compute
# tensors at the root separately by default.
A = tvm.placeholder((m, ), name='A')
A1 = tvm.compute((m, ), lambda i: A[i]+1, name='A1')
T = tvm.compute((m, ), lambda i: A1[i]*2, name='T')

s = tvm.create_schedule(T.op)
PrintSchedule(s)

######################################################################
# :code:`compute_at` can move computation of `A1` into the first axis
# of computation of `T`.
A = tvm.placeholder((m, ), name='A')
A1 = tvm.compute((m, ), lambda i: A[i]+1, name='A1')
T = tvm.compute((m, ), lambda i: A1[i]*2, name='T')

s = tvm.create_schedule(T.op)
s[A1].compute_at(s[T], T.op.axis[0])
PrintSchedule(s)

######################################################################
# :code:`compute_inline` can mark one stage as inline, then the body of
# computation will be expanded and inserted at the address where the
# tensor is required.
A = tvm.placeholder((m, ), name='A')
A1 = tvm.compute((m, ), lambda i: A[i]+1, name='A1')
T = tvm.compute((m, ), lambda i: A1[i]*2, name='T')
s = tvm.create_schedule(T.op)

s = tvm.create_schedule(T.op)
s[A1].compute_inline()
PrintSchedule(s)

######################################################################
# :code:`compute_root` can move computation of one stage to the root.
A = tvm.placeholder((m, ), name='A')
A1 = tvm.compute((m, ), lambda i: A[i]+1, name='A1')
T = tvm.compute((m, ), lambda i: A1[i]*2, name='T')
s = tvm.create_schedule(T.op)

s = tvm.create_schedule(T.op)
s[A1].compute_at(s[T], T.op.axis[0])
s[A1].compute_root()
PrintSchedule(s)

######################################################################
# Here is one practical example which implements GEMM with TVM.
def tvm_gemm():
    # graph
    nn = 1024
    n = tvm.var('n')
    n = tvm.convert(nn)
    m = n
    l = n
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((m, l), name='B')
    k = tvm.reduce_axis((0, l), name='k')
    C = tvm.compute(
        (n, m),
        lambda ii, jj: tvm.sum(A[ii, k] * B[jj, k], axis=k),
        name='CC')
    s = tvm.create_schedule(C.op)
    xtile, ytile = 32, 32
    scale = 8
    num_thread = 8
    block_factor = scale * num_thread
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_y = tvm.thread_axis("threadIdx.y")

    CC = s.cache_write(C, "local")
    AA = s.cache_read(A, "shared", [CC])
    BB = s.cache_read(B, "shared", [CC])
    by, yi = s[C].split(C.op.axis[0], factor=block_factor)
    bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
    s[C].reorder(by, bx, yi, xi)
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    ty, yi = s[C].split(yi, nparts=num_thread)
    tx, xi = s[C].split(xi, nparts=num_thread)
    s[C].reorder(ty, tx, yi, xi)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    yo, xo = CC.op.axis
    s[CC].reorder(k, yo, xo)


    s[CC].compute_at(s[C], tx)
    s[AA].compute_at(s[CC], k)
    s[BB].compute_at(s[CC], k)

    ty, xi = s[AA].split(s[AA].op.axis[0], nparts=num_thread)
    tx, xi = s[AA].split(xi, nparts=num_thread)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)

    ty, xi = s[BB].split(s[BB].op.axis[0], nparts=num_thread)
    tx, xi = s[BB].split(xi, nparts=num_thread)
    s[BB].bind(ty, thread_y)
    s[BB].bind(tx, thread_x)

    s.normalize()

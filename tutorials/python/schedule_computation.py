"""
Schedule Primitives in TVM
=================================
**Author**: `Ziheng Jiang <https://github.com/ZihengJiang>`_

TVM is a domain specific language for efficient kernel construction.

In this tutorial, we will show you how to schedule the computation by
primitives provided by TVM.
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
def print_schedule(s, ops):
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    print(stmt)

######################################################################
# A schedule can be created from a list of ops, by default the
# schedule compute tensor in a serial manner in a row-major order.

# declare a matrix element-wise multiply
A = tvm.placeholder((m, n), name='A')
B = tvm.placeholder((m, n), name='B')
C = tvm.compute((m, n), lambda i, j: A[i, j] * B[i, j], name='C')

s = tvm.create_schedule([C.op])
print_schedule(s)

######################################################################
# One schedule is composed by multiple stages, and one
# **Stage** represents schedule for one operation. We provide various
# methods to schedule every stage.

######################################################################
# split
# --------------------------
# :code:`split` can split a specified axis into two axises by
# :code:`factor`.
A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i]*2, name='B')

s = tvm.create_schedule(B.op)
xo, xi = s[B].split(B.op.axis[0], factor=32)
print_schedule(s)

######################################################################
# You can also split a axis by :code:`nparts`, which splits the axis
# contrary with :code:`factor`.
A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i], name='B')

s = tvm.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], nparts=32)
print_schedule(s)

######################################################################
# tile
# --------------------------
# :code:`tile` help you execute the computation tile by tile over two
# axises.
A = tvm.placeholder((m, n), name='A')
B = tvm.compute((m, n), lambda i, j: A[i, j], name='B')

s = tvm.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
print_schedule(s)

######################################################################
# fuse
# --------------------------
# :code:`fuse` can fuse two consecutive axises of one computation.
A = tvm.placeholder((m, n), name='A')
B = tvm.compute((m, n), lambda i, j: A[i, j], name='B')

s = tvm.create_schedule(B.op)
# tile to four axises first: (i.outer, j.outer, i.inner, j.inner)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# then fuse (i.inner, j.inner) into one axis: (i.inner.j.inner.fused)
fused = s[B].fuse(yi, xi)
print_schedule(s)

######################################################################
# reorder
# --------------------------
# :code:`reorder` can reorder the axises in the specified order.
A = tvm.placeholder((m, n), name='A')
B = tvm.compute((m, n), lambda i, j: A[i, j], name='B')

s = tvm.create_schedule(B.op)
# tile to four axises first: (i.outer, j.outer, i.inner, j.inner)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# then reorder the axises: (i.inner, j.outer, i.outer, j.inner)
s[B].reorder(xi, yo, xo, yi)
print_schedule(s)

######################################################################
# bind
# --------------------------
# :code:`bind` can bind a specified axis with a thread axis, often used
# in gpu programming.
A = tvm.placeholder((n,), name='A')
B = tvm.compute(A.shape, lambda i: A[i] * 2, name='B')

s = tvm.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], factor=64)
s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
print_schedule(s)

######################################################################
# compute_at
# --------------------------
# For a schedule consists of multiple operators, tvm will compute
# tensors at the root separately by default.
A = tvm.placeholder((m, ), name='A')
B = tvm.compute((m, ), lambda i: A[i]+1, name='B')
C = tvm.compute((m, ), lambda i: B[i]*2, name='C')

s = tvm.create_schedule(C.op)
print_schedule(s)

######################################################################
# :code:`compute_at` can move computation of `B` into the first axis
# of computation of `C`.
A = tvm.placeholder((m, ), name='A')
B = tvm.compute((m, ), lambda i: A[i]+1, name='B')
C = tvm.compute((m, ), lambda i: B[i]*2, name='C')

s = tvm.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
print_schedule(s)

######################################################################
# compute_inline
# --------------------------
# :code:`compute_inline` can mark one stage as inline, then the body of
# computation will be expanded and inserted at the address where the
# tensor is required.
A = tvm.placeholder((m, ), name='A')
B = tvm.compute((m, ), lambda i: A[i]+1, name='B')
C = tvm.compute((m, ), lambda i: B[i]*2, name='C')

s = tvm.create_schedule(C.op)
s[B].compute_inline()
print_schedule(s)

######################################################################
# compute_root
# --------------------------
# :code:`compute_root` can move computation of one stage to the root.
A = tvm.placeholder((m, ), name='A')
B = tvm.compute((m, ), lambda i: A[i]+1, name='B')
C = tvm.compute((m, ), lambda i: B[i]*2, name='C')
s = tvm.create_schedule(C.op)

s = tvm.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
s[B].compute_root()
print_schedule(s)

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

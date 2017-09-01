"""
Schedule Primitives in TVM
==========================
**Author**: `Ziheng Jiang <https://github.com/ZihengJiang>`_

TVM is a domain specific language for efficient kernel construction.

In this tutorial, we will show you how to schedule the computation by
various primitives provided by TVM.
"""
from __future__ import absolute_import, print_function

import tvm
import numpy as np

######################################################################
#
# There often exist several methods to compute the same result,
# however, different methods will result in different locality and
# performance. So TVM asks user to provide how to execute the
# computation called **Schedule**.
#
# A **Schedule** is a set of transformation of computation that
# transforms the loop of computations in the program.
#

# declare some variables for use later
n = tvm.var('n')
m = tvm.var('m')

######################################################################
# A schedule can be created from a list of ops, by default the
# schedule computes tensor in a serial manner in a row-major order.

# declare a matrix element-wise multiply
A = tvm.placeholder((m, n), name='A')
B = tvm.placeholder((m, n), name='B')
C = tvm.compute((m, n), lambda i, j: A[i, j] * B[i, j], name='C')

s = tvm.create_schedule([C.op])
# lower will transform the computation from definition to the real
# callable function. With argument `simple_mode=True`, it will
# return you a readable C like statement, we use it here to print the
# schedule result.
print(tvm.lower(s, [A, B, C], simple_mode=True))

######################################################################
# One schedule is composed by multiple stages, and one
# **Stage** represents schedule for one operation. We provide various
# methods to schedule every stage.

######################################################################
# split
# -----
# :code:`split` can split a specified axis into two axises by
# :code:`factor`.
A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i]*2, name='B')

s = tvm.create_schedule(B.op)
xo, xi = s[B].split(B.op.axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))

######################################################################
# You can also split a axis by :code:`nparts`, which splits the axis
# contrary with :code:`factor`.
A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i], name='B')

s = tvm.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], nparts=32)
print(tvm.lower(s, [A, B], simple_mode=True))

######################################################################
# tile
# ----
# :code:`tile` help you execute the computation tile by tile over two
# axises.
A = tvm.placeholder((m, n), name='A')
B = tvm.compute((m, n), lambda i, j: A[i, j], name='B')

s = tvm.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
print(tvm.lower(s, [A, B], simple_mode=True))

######################################################################
# fuse
# ----
# :code:`fuse` can fuse two consecutive axises of one computation.
A = tvm.placeholder((m, n), name='A')
B = tvm.compute((m, n), lambda i, j: A[i, j], name='B')

s = tvm.create_schedule(B.op)
# tile to four axises first: (i.outer, j.outer, i.inner, j.inner)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# then fuse (i.inner, j.inner) into one axis: (i.inner.j.inner.fused)
fused = s[B].fuse(xi, yi)
print(tvm.lower(s, [A, B], simple_mode=True))

######################################################################
# reorder
# -------
# :code:`reorder` can reorder the axises in the specified order.
A = tvm.placeholder((m, n), name='A')
B = tvm.compute((m, n), lambda i, j: A[i, j], name='B')

s = tvm.create_schedule(B.op)
# tile to four axises first: (i.outer, j.outer, i.inner, j.inner)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# then reorder the axises: (i.inner, j.outer, i.outer, j.inner)
s[B].reorder(xi, yo, xo, yi)
print(tvm.lower(s, [A, B], simple_mode=True))

######################################################################
# bind
# ----
# :code:`bind` can bind a specified axis with a thread axis, often used
# in gpu programming.
A = tvm.placeholder((n,), name='A')
B = tvm.compute(A.shape, lambda i: A[i] * 2, name='B')

s = tvm.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], factor=64)
s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
print(tvm.lower(s, [A, B], simple_mode=True))

######################################################################
# compute_at
# ----------
# For a schedule consists of multiple operators, tvm will compute
# tensors at the root separately by default.
A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i]+1, name='B')
C = tvm.compute((m,), lambda i: B[i]*2, name='C')

s = tvm.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))

######################################################################
# :code:`compute_at` can move computation of `B` into the first axis
# of computation of `C`.
A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i]+1, name='B')
C = tvm.compute((m,), lambda i: B[i]*2, name='C')

s = tvm.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
print(tvm.lower(s, [A, B, C], simple_mode=True))

######################################################################
# compute_inline
# --------------
# :code:`compute_inline` can mark one stage as inline, then the body of
# computation will be expanded and inserted at the address where the
# tensor is required.
A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i]+1, name='B')
C = tvm.compute((m,), lambda i: B[i]*2, name='C')

s = tvm.create_schedule(C.op)
s[B].compute_inline()
print(tvm.lower(s, [A, B, C], simple_mode=True))

######################################################################
# compute_root
# ------------
# :code:`compute_root` can move computation of one stage to the root.
A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i]+1, name='B')
C = tvm.compute((m,), lambda i: B[i]*2, name='C')

s = tvm.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
s[B].compute_root()
print(tvm.lower(s, [A, B, C], simple_mode=True))

######################################################################
# Summary
# -------
# This tutorial provides an introduction to schedule primitives in
# tvm, which permits users schedule the computation easily and
# flexibly.
#
# In order to get a good performance kernel implementation, the
# general workflow often is:
#
# - Describe your computation via series of operations.
# - Try to schedule the computation with primitives.
# - Compile and run to see the performance difference.
# - Adjust your schedule according the running result.

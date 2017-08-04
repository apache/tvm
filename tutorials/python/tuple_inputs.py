"""
Compute and Reduce with Tuple Inputs
=======================================
**Author**: `Ziheng Jiang <https://github.com/ZihengJiang>`_

Often we want to compute multiple outputs with the same shape within
a single loop or perform reduction that involves multiple values like
:code:`argmax`. These problems can be addressed by tuple inputs.

In this tutorial, we will introduce the usage of tuple inputs in TVM.
"""
from __future__ import absolute_import, print_function

import tvm
import numpy as np

######################################################################
# Describe Batchwise Computation
# ------------------------------
# For operators which have the same shape, we can put them together as
# the inputs of :any:`tvm.compute`, if we wish they can be scheduled
# together in the next schedule procedure.
#
n = tvm.var("n")
m = tvm.var("m")
A0 = tvm.placeholder((m, n), name='A0')
A1 = tvm.placeholder((m, n), name='A1')
B0, B1 = tvm.compute((m, n), lambda i, j: (A0[i, j] + 2, A1[i, j] * 3), name='B')

# The generated IR code would be:
s = tvm.create_schedule(B0.op)
print(tvm.lower(s, [A0, A1, B0, B1], simple_mode=True))

######################################################################
# .. _reduction-with-tuple-inputs:
#
# Describe Reduction with Collaborative Inputs
# --------------------------------------------
# Sometimes, we require multiple inputs to express some reduction
# operators, and the inputs will collaborate together, e.g. :code:`argmax`.
# In the reduction procedure, :code:`argmax` need to compare the value of
# operands, also need to keep the index of operand. It can be expressed
# with :any:`comm_reducer` as below:

# x and y are the operands of reduction, both of them is a tuple of index
# and value.
def fcombine(x, y):
    lhs = tvm.select((x[1] >= y[1]), x[0], y[0])
    rhs = tvm.select((x[1] >= y[1]), x[1], y[1])
    return lhs, rhs

# our identity element also need to be a tuple, so `fidentity` accepts
# two types as inputs.
def fidentity(t0, t1):
    return tvm.const(-1, t0), tvm.min_value(t1)

argmax = tvm.comm_reducer(fcombine, fidentity, name='argmax')

# describe the reduction computation
m = tvm.var('m')
n = tvm.var('n')
idx = tvm.placeholder((m, n), name='idx', dtype='int32')
val = tvm.placeholder((m, n), name='val', dtype='int32')
k = tvm.reduce_axis((0, n), 'k')
T0, T1 = tvm.compute((m, ), lambda i: argmax((idx[i, k], val[i, k]), axis=k), name='T')

# the generated IR code would be:
s = tvm.create_schedule(T0.op)
print(tvm.lower(s, [idx, val, T0, T1], simple_mode=True))

######################################################################
# .. note::
#
#   For ones who are not familiar with reduction, please refer to
#   :ref:`general-reduction`.

######################################################################
# Schedule Operation with Tuple Inputs
# ------------------------------------
# It is worth mentioning that although you will get multiple outputs
# with one batch operation, but they can only be scheduled together
# in terms of operation.

n = tvm.var("n")
m = tvm.var("m")
A0 = tvm.placeholder((m, n), name='A0')
B0, B1 = tvm.compute((m, n), lambda i, j: (A0[i, j] + 2, A0[i, j] * 3), name='B')
A1 = tvm.placeholder((m, n), name='A1')
C = tvm.compute((m, n), lambda i, j: A1[i, j] + B0[i, j], name='C')

s = tvm.create_schedule(C.op)
s[B0].compute_at(s[C], C.op.axis[0])
# as you can see in the below generated IR code:
print(tvm.lower(s, [A0, A1, C], simple_mode=True))

######################################################################
# Summary
# -------
# This tutorial introduces the usage of tuple inputs operation.
#
# - Describe normal batchwise computation.
# - Describe reduction operation with tuple inputs.
# - Notice that you can only schedule computation in terms of operation instead of tensor.

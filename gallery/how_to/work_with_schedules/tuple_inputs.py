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
from tvm import te
import numpy as np

######################################################################
# Describe Batchwise Computation
# ------------------------------
# For operators which have the same shape, we can put them together as
# the inputs of :any:`te.compute`, if we want them to be scheduled
# together in the next schedule procedure.
#
n = te.var("n")
m = te.var("m")
A0 = te.placeholder((m, n), name="A0")
A1 = te.placeholder((m, n), name="A1")
B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] + 2, A1[i, j] * 3), name="B")

# The generated IR code would be:
s = te.create_schedule(B0.op)
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
# with :py:func:`te.comm_reducer` as below:

# x and y are the operands of reduction, both of them is a tuple of index
# and value.
def fcombine(x, y):
    lhs = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
    rhs = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
    return lhs, rhs


# our identity element also need to be a tuple, so `fidentity` accepts
# two types as inputs.
def fidentity(t0, t1):
    return tvm.tir.const(-1, t0), tvm.te.min_value(t1)


argmax = te.comm_reducer(fcombine, fidentity, name="argmax")

# describe the reduction computation
m = te.var("m")
n = te.var("n")
idx = te.placeholder((m, n), name="idx", dtype="int32")
val = te.placeholder((m, n), name="val", dtype="int32")
k = te.reduce_axis((0, n), "k")
T0, T1 = te.compute((m,), lambda i: argmax((idx[i, k], val[i, k]), axis=k), name="T")

# the generated IR code would be:
s = te.create_schedule(T0.op)
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

n = te.var("n")
m = te.var("m")
A0 = te.placeholder((m, n), name="A0")
B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] + 2, A0[i, j] * 3), name="B")
A1 = te.placeholder((m, n), name="A1")
C = te.compute((m, n), lambda i, j: A1[i, j] + B0[i, j], name="C")

s = te.create_schedule(C.op)
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

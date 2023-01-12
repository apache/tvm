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
Reduction
=========
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

This is an introduction material on how to do reduction in TVM.
Associative reduction operators like sum/max/min are typical
construction blocks of linear algebra operations.

In this tutorial, we will demonstrate how to do reduction in TVM.
"""
from __future__ import absolute_import, print_function


# sphinx_gallery_start_ignore
# sphinx_gallery_requires_cuda = True
# sphinx_gallery_end_ignore
import tvm
import tvm.testing
from tvm import te
import numpy as np

######################################################################
# Describe Sum of Rows
# --------------------
# Assume we want to compute sum of rows as our example.
# In numpy semantics this can be written as :code:`B = numpy.sum(A, axis=1)`
#
# The following lines describe the row sum operation.
# To create a reduction formula, we declare a reduction axis using
# :any:`te.reduce_axis`. :any:`te.reduce_axis` takes in the range of reductions.
# :any:`te.sum` takes in the expression to be reduced as well as the reduction
# axis and compute the sum of value over all k in the declared range.
#
# The equivalent C code is as follows:
#
# .. code-block:: c
#
#   for (int i = 0; i < n; ++i) {
#     B[i] = 0;
#     for (int k = 0; k < m; ++k) {
#       B[i] = B[i] + A[i][k];
#     }
#   }
#
n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), "k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")

######################################################################
# Schedule the Reduction
# ----------------------
# There are several ways to schedule a reduction.
# Before doing anything, let us print out the IR code of default schedule.
#
s = te.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))

######################################################################
# You can find that the IR code is quite like the C code.
# The reduction axis is similar to a normal axis, it can be splitted.
#
# In the following code we split both the row axis of B as well
# axis by different factors. The result is a nested reduction.
#
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
xo, xi = s[B].split(B.op.axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))

######################################################################
# If we are building a GPU kernel, we can bind the rows of B to GPU threads.
s[B].bind(xo, te.thread_axis("blockIdx.x"))
s[B].bind(xi, te.thread_axis("threadIdx.x"))
print(tvm.lower(s, [A, B], simple_mode=True))

######################################################################
# Reduction Factoring and Parallelization
# ---------------------------------------
# One problem of building a reduction is that we cannot simply
# parallelize over the reduction axis. We need to divide the computation
# of the reduction, store the local reduction result in a temporal array
# before doing a reduction over the temp array.
#
# The rfactor primitive does such rewrite of the computation.
# In the following schedule, the result of B is written to a temporary
# result B.rf. The factored dimension becomes the first dimension of B.rf.
#
s = te.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
print(tvm.lower(s, [A, B], simple_mode=True))

######################################################################
# The scheduled operator of B also get rewritten to be sum over
# the first axis of reduced result of B.f
#
print(s[B].op.body)

######################################################################
# Cross Thread Reduction
# ----------------------
# We can now parallelize over the factored axis.
# Here the reduction axis of B is marked to be a thread.
# TVM allows reduction axis to be marked as thread if it is the only
# axis in reduction and cross thread reduction is possible in the device.
#
# This is indeed the case after the factoring.
# We can directly compute BF at the reduction axis as well.
# The final generated kernel will divide the rows by blockIdx.x and threadIdx.y
# columns by threadIdx.x and finally do a cross thread reduction over threadIdx.x
#
xo, xi = s[B].split(s[B].op.axis[0], factor=32)
s[B].bind(xo, te.thread_axis("blockIdx.x"))
s[B].bind(xi, te.thread_axis("threadIdx.y"))
tx = te.thread_axis("threadIdx.x")
s[B].bind(s[B].op.reduce_axis[0], tx)
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
s[B].set_store_predicate(tx.var.equal(0))
fcuda = tvm.build(s, [A, B], "cuda")
print(fcuda.imported_modules[0].get_source())

######################################################################
# Verify the correctness of result kernel by comparing it to numpy.
#
nn = 128
dev = tvm.cuda(0)
a = tvm.nd.array(np.random.uniform(size=(nn, nn)).astype(A.dtype), dev)
b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), dev)
fcuda(a, b)
tvm.testing.assert_allclose(b.numpy(), np.sum(a.numpy(), axis=1), rtol=1e-4)

######################################################################
# Describe Convolution via 2D Reduction
# -------------------------------------
# In TVM, we can describe convolution via 2D reduction in a simple way.
# Here is an example for 2D convolution with filter size = [3, 3] and strides = [1, 1].
#
n = te.var("n")
Input = te.placeholder((n, n), name="Input")
Filter = te.placeholder((3, 3), name="Filter")
di = te.reduce_axis((0, 3), name="di")
dj = te.reduce_axis((0, 3), name="dj")
Output = te.compute(
    (n - 2, n - 2),
    lambda i, j: te.sum(Input[i + di, j + dj] * Filter[di, dj], axis=[di, dj]),
    name="Output",
)
s = te.create_schedule(Output.op)
print(tvm.lower(s, [Input, Filter, Output], simple_mode=True))

######################################################################
# .. _general-reduction:
#
# Define General Commutative Reduction Operation
# ----------------------------------------------
# Besides the built-in reduction operations like :any:`te.sum`,
# :any:`tvm.te.min` and :any:`tvm.te.max`, you can also define your
# commutative reduction operation by :any:`te.comm_reducer`.
#

n = te.var("n")
m = te.var("m")
product = te.comm_reducer(lambda x, y: x * y, lambda t: tvm.tir.const(1, dtype=t), name="product")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), name="k")
B = te.compute((n,), lambda i: product(A[i, k], axis=k), name="B")

######################################################################
# .. note::
#
#   Sometimes we would like to perform reduction that involves multiple
#   values like :code:`argmax`, which can be done by tuple inputs.
#   See :ref:`reduction-with-tuple-inputs` for more detail.

######################################################################
# Summary
# -------
# This tutorial provides a walk through of reduction schedule.
#
# - Describe reduction with reduce_axis.
# - Use rfactor to factor out axis if we need parallelism.
# - Define new reduction operation by :any:`te.comm_reducer`

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
.. _tutorial-topi:

Introduction to TOPI
====================
**Author**: `Ehsan M. Kermani <https://github.com/ehsanmok>`_

This is an introductory tutorial to TVM Operator Inventory (TOPI).
TOPI provides numpy-style generic operations and schedules with higher abstractions than TVM.
In this tutorial, we will see how TOPI can save us from writing boilerplate code in TVM.
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_requires_cuda = True
# sphinx_gallery_end_ignore
import tvm
import tvm.testing
from tvm import te
from tvm import topi
import numpy as np

######################################################################
# Basic example
# -------------
# Let's revisit the sum of rows operation (equivalent to :code:`B = numpy.sum(A, axis=1)`') \
# To compute the sum of rows of a two dimensional TVM tensor A, we should
# specify the symbolic operation as well as schedule as follows
#
n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), "k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
s = te.create_schedule(B.op)

######################################################################
# and to examine the IR code in human readable format, we can do
#
print(tvm.lower(s, [A], simple_mode=True))

######################################################################
# However, for such a common operation we had to define the reduce axis ourselves as well as explicit computation with
# :code:`te.compute`. Imagine for more complicated operations how much details we need to provide.
# Fortunately, we can replace those two lines with simple :code:`topi.sum` much like :code:`numpy.sum`
#
C = topi.sum(A, axis=1)
ts = te.create_schedule(C.op)
print(tvm.lower(ts, [A], simple_mode=True))

######################################################################
# Numpy-style operator overloading
# --------------------------------
# We can add two tensors using :code:`topi.broadcast_add` that have correct (broadcastable with specific) shapes.
# Even shorter, TOPI provides operator overloading for such common operations. For example,
#
x, y = 100, 10
a = te.placeholder((x, y, y), name="a")
b = te.placeholder((y, y), name="b")
c = a + b  # same as topi.broadcast_add
d = a * b  # same as topi.broadcast_mul

######################################################################
# Overloaded with the same syntax, TOPI handles broadcasting a primitive (`int`, `float`) to a tensor :code:`d - 3.14`.

######################################################################
# Generic schedules and fusing operations
# ---------------------------------------
# Up to now, we have seen an example of how TOPI can save us from writing explicit computations in lower level API.
# But it doesn't stop here. Still we did the scheduling as before. TOPI also provides higher level
# scheduling recipes depending on a given context. For example, for CUDA,
# we can schedule the following series of operations ending with :code:`topi.sum` using only
# :code:`topi.generic.schedule_reduce`
#
e = topi.elemwise_sum([c, d])
f = e / 2.0
g = topi.sum(f)
with tvm.target.cuda():
    sg = topi.cuda.schedule_reduce(g)
    print(tvm.lower(sg, [a, b], simple_mode=True))

######################################################################
# As you can see, scheduled stages of computation have been accumulated and we can examine them by
#
print(sg.stages)

######################################################################
# We can test the correctness by comparing with :code:`numpy` result as follows
#
func = tvm.build(sg, [a, b, g], "cuda")
dev = tvm.cuda(0)
a_np = np.random.uniform(size=(x, y, y)).astype(a.dtype)
b_np = np.random.uniform(size=(y, y)).astype(b.dtype)
g_np = np.sum(np.add(a_np + b_np, a_np * b_np) / 2.0)
a_nd = tvm.nd.array(a_np, dev)
b_nd = tvm.nd.array(b_np, dev)
g_nd = tvm.nd.array(np.zeros(g_np.shape, dtype=g_np.dtype), dev)
func(a_nd, b_nd, g_nd)
tvm.testing.assert_allclose(g_nd.numpy(), g_np, rtol=1e-5)

######################################################################
# TOPI also provides common neural nets operations such as _softmax_ with optimized schedule
#
tarray = te.placeholder((512, 512), name="tarray")
softmax_topi = topi.nn.softmax(tarray)
with tvm.target.Target("cuda"):
    sst = topi.cuda.schedule_softmax(softmax_topi)
    print(tvm.lower(sst, [tarray], simple_mode=True))

######################################################################
# Fusing convolutions
# -------------------
# We can fuse :code:`topi.nn.conv2d` and :code:`topi.nn.relu` together.
#
# .. note::
#
#    TOPI functions are all generic functions. They have different implementations
#    for different backends to optimize for performance.
#    For each backend, it is necessary to call them under a target scope for both
#    compute declaration and schedule. TVM will choose the right function to call with
#    the target information.

data = te.placeholder((1, 3, 224, 224))
kernel = te.placeholder((10, 3, 5, 5))

with tvm.target.Target("cuda"):
    conv = topi.cuda.conv2d_nchw(data, kernel, 1, 2, 1)
    out = topi.nn.relu(conv)
    sconv = topi.cuda.schedule_conv2d_nchw([out])
    print(tvm.lower(sconv, [data, kernel], simple_mode=True))

######################################################################
# Summary
# -------
# In this tutorial, we have seen
#
# - How to use TOPI API for common operations with numpy-style operators.
# - How TOPI facilitates generic schedules and operator fusion for a context, to generate optimized kernel codes.

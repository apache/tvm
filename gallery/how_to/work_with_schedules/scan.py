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
Scan and Recurrent Kernel
=========================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

This is an introduction material on how to do recurrent computing in TVM.
Recurrent computing is a typical pattern in neural networks.
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
# TVM supports a scan operator to describe symbolic loop.
# The following scan op computes cumsum over columns of X.
#
# The scan is carried over the highest dimension of the tensor.
# :code:`s_state` is a placeholder that describes the transition state of the scan.
# :code:`s_init` describes how we can initialize the first k timesteps.
# Here since s_init's first dimension is 1, it describes how we initialize
# The state at first timestep.
#
# :code:`s_update` describes how to update the value at timestep t. The update
# value can refer back to the values of previous timestep via state placeholder.
# Note that while it is invalid to refer to :code:`s_state` at current or later timestep.
#
# The scan takes in state placeholder, initial value and update description.
# It is also recommended(although not necessary) to list the inputs to the scan cell.
# The result of the scan is a tensor, giving the result of :code:`s_state` after the
# update over the time domain.
#
m = te.var("m")
n = te.var("n")
X = te.placeholder((m, n), name="X")
s_state = te.placeholder((m, n))
s_init = te.compute((1, n), lambda _, i: X[0, i])
s_update = te.compute((m, n), lambda t, i: s_state[t - 1, i] + X[t, i])
s_scan = tvm.te.scan(s_init, s_update, s_state, inputs=[X])

######################################################################
# Schedule the Scan Cell
# ----------------------
# We can schedule the body of the scan by scheduling the update and
# init part separately. Note that it is invalid to schedule the
# first iteration dimension of the update part.
# To split on the time iteration, user can schedule on scan_op.scan_axis instead.
#
s = te.create_schedule(s_scan.op)
num_thread = 256
block_x = te.thread_axis("blockIdx.x")
thread_x = te.thread_axis("threadIdx.x")
xo, xi = s[s_init].split(s_init.op.axis[1], factor=num_thread)
s[s_init].bind(xo, block_x)
s[s_init].bind(xi, thread_x)
xo, xi = s[s_update].split(s_update.op.axis[1], factor=num_thread)
s[s_update].bind(xo, block_x)
s[s_update].bind(xi, thread_x)
print(tvm.lower(s, [X, s_scan], simple_mode=True))

######################################################################
# Build and Verify
# ----------------
# We can build the scan kernel like other TVM kernels, here we use
# numpy to verify the correctness of the result.
#
fscan = tvm.build(s, [X, s_scan], "cuda", name="myscan")
dev = tvm.cuda(0)
n = 1024
m = 10
a_np = np.random.uniform(size=(m, n)).astype(s_scan.dtype)
a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(np.zeros((m, n), dtype=s_scan.dtype), dev)
fscan(a, b)
tvm.testing.assert_allclose(b.numpy(), np.cumsum(a_np, axis=0))

######################################################################
# Multi-Stage Scan Cell
# ---------------------
# In the above example we described the scan cell using one Tensor
# computation stage in s_update. It is possible to use multiple
# Tensor stages in the scan cell.
#
# The following lines demonstrate a scan with two stage operations
# in the scan cell.
#
m = te.var("m")
n = te.var("n")
X = te.placeholder((m, n), name="X")
s_state = te.placeholder((m, n))
s_init = te.compute((1, n), lambda _, i: X[0, i])
s_update_s1 = te.compute((m, n), lambda t, i: s_state[t - 1, i] * 2, name="s1")
s_update_s2 = te.compute((m, n), lambda t, i: s_update_s1[t, i] + X[t, i], name="s2")
s_scan = tvm.te.scan(s_init, s_update_s2, s_state, inputs=[X])

######################################################################
# These intermediate tensors can also be scheduled normally.
# To ensure correctness, TVM creates a group constraint to forbid
# the body of scan to be compute_at locations outside the scan loop.
#
s = te.create_schedule(s_scan.op)
xo, xi = s[s_update_s2].split(s_update_s2.op.axis[1], factor=32)
s[s_update_s1].compute_at(s[s_update_s2], xo)
print(tvm.lower(s, [X, s_scan], simple_mode=True))

######################################################################
# Multiple States
# ---------------
# For complicated applications like RNN, we might need more than one
# recurrent state. Scan support multiple recurrent states.
# The following example demonstrates how we can build recurrence with two states.
#
m = te.var("m")
n = te.var("n")
l = te.var("l")
X = te.placeholder((m, n), name="X")
s_state1 = te.placeholder((m, n))
s_state2 = te.placeholder((m, l))
s_init1 = te.compute((1, n), lambda _, i: X[0, i])
s_init2 = te.compute((1, l), lambda _, i: 0.0)
s_update1 = te.compute((m, n), lambda t, i: s_state1[t - 1, i] + X[t, i])
s_update2 = te.compute((m, l), lambda t, i: s_state2[t - 1, i] + s_state1[t - 1, 0])
s_scan1, s_scan2 = tvm.te.scan(
    [s_init1, s_init2], [s_update1, s_update2], [s_state1, s_state2], inputs=[X]
)
s = te.create_schedule(s_scan1.op)
print(tvm.lower(s, [X, s_scan1, s_scan2], simple_mode=True))

######################################################################
# Summary
# -------
# This tutorial provides a walk through of scan primitive.
#
# - Describe scan with init and update.
# - Schedule the scan cells as normal schedule.
# - For complicated workload, use multiple states and steps in scan cell.

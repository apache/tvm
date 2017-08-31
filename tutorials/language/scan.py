"""
Scan and Recurrent Kernel
=========================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

This is an introduction material on how to do recurrent computing in TVM.
Recurrent computing is a typical pattern in neural networks.
"""
from __future__ import absolute_import, print_function

import tvm
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
m = tvm.var("m")
n = tvm.var("n")
X = tvm.placeholder((m, n), name="X")
s_state = tvm.placeholder((m, n))
s_init = tvm.compute((1, n), lambda _, i: X[0, i])
s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
s_scan = tvm.scan(s_init, s_update, s_state, inputs=[X])

######################################################################
# Schedule the Scan Cell
# ----------------------
# We can schedule the body of the scan by scheduling the update and
# init part seperately. Note that it is invalid to schedule the
# first iteration dimension of the update part.
# To split on the time iteration, user can schedule on scan_op.scan_axis instead.
#
s = tvm.create_schedule(s_scan.op)
num_thread = 256
block_x = tvm.thread_axis("blockIdx.x")
thread_x = tvm.thread_axis("threadIdx.x")
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
# We can build the scan kernel like other tvm kernels, here we use
# numpy to verify the correctness of the result.
#
fscan = tvm.build(s, [X, s_scan], "cuda", name="myscan")
ctx = tvm.gpu(0)
n = 1024
m = 10
a_np = np.random.uniform(size=(m, n)).astype(s_scan.dtype)
a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(np.zeros((m, n), dtype=s_scan.dtype), ctx)
fscan(a, b)
np.testing.assert_allclose(b.asnumpy(), np.cumsum(a_np, axis=0))

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
m = tvm.var("m")
n = tvm.var("n")
X = tvm.placeholder((m, n), name="X")
s_state = tvm.placeholder((m, n))
s_init = tvm.compute((1, n), lambda _, i: X[0, i])
s_update_s1 = tvm.compute((m, n), lambda t, i: s_state[t-1, i] * 2, name="s1")
s_update_s2 = tvm.compute((m, n), lambda t, i: s_update_s1[t, i] + X[t, i], name="s2")
s_scan = tvm.scan(s_init, s_update_s2, s_state, inputs=[X])

######################################################################
# These intermediate tensors can also be scheduled normally.
# To ensure correctness, TVM creates a group constraint to forbid
# the body of scan to be compute_at locations outside the scan loop.
#
s = tvm.create_schedule(s_scan.op)
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
m = tvm.var("m")
n = tvm.var("n")
l = tvm.var("l")
X = tvm.placeholder((m, n), name="X")
s_state1 = tvm.placeholder((m, n))
s_state2 = tvm.placeholder((m, l))
s_init1 = tvm.compute((1, n), lambda _, i: X[0, i])
s_init2 = tvm.compute((1, l), lambda _, i: 0.0)
s_update1 = tvm.compute((m, n), lambda t, i: s_state1[t-1, i] + X[t, i])
s_update2 = tvm.compute((m, l), lambda t, i: s_state2[t-1, i] + s_state1[t-1, 0])
s_scan1, s_scan2 = tvm.scan([s_init1, s_init2],
                            [s_update1, s_update2],
                            [s_state1, s_state2], inputs=[X])
s = tvm.create_schedule(s_scan1.op)
print(tvm.lower(s, [X, s_scan1, s_scan2], simple_mode=True))

######################################################################
# Summary
# -------
# This tutorial provides a walk through of scan primitive.
#
# - Describe scan with init and update.
# - Schedule the scan cells as normal schedule.
# - For complicated workload, use multiple states and steps in scan cell.

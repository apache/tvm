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
.. _vta-mat-mult-opt:

Matrix Multiply Blocking
========================
**Author**: `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

This tutorial provides an overview on how to use TVM to map matrix
multiplication efficiently on the VTA design.
We recommend covering the :ref:`basic-mat-mult` tutorial first.

In this tutorial, we will demonstrate TVM schedule optimizations to break large
neural network operators down onto smaller blocks to achieve computation within
limited hardware accelerator resources.
"""

######################################################################
# RPC Setup
# ---------
# We start by programming the Pynq's FPGA and building its RPC runtime.

from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import vta
import numpy as np
from tvm import rpc
from tvm.contrib import utils
from vta.testing import simulator

# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()

# We read the Pynq RPC host IP address and port number from the OS environment
host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
port = int(os.environ.get("VTA_RPC_PORT", "9091"))

# We configure both the bitstream and the runtime system on the Pynq
# to match the VTA configuration specified by the vta_config.json file.
if env.TARGET == "pynq":

    # Make sure that TVM was compiled with RPC=1
    assert tvm.runtime.enabled("rpc")
    remote = rpc.connect(host, port)

    # Reconfigure the JIT runtime
    vta.reconfig_runtime(remote)

    # Program the FPGA with a pre-compiled VTA bitstream.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    vta.program_fpga(remote, bitstream=None)

# In simulation mode, host the RPC server locally.
elif env.TARGET in ["sim", "tsim"]:
    remote = rpc.LocalSession()

######################################################################
# Computation Declaration
# -----------------------
# As a first step, we need to describe our matrix multiplication computation.
# We define the matrix multiplication as the computation one would find in a
# fully connected layer, defined by its batch size, input channels, and output
# channels.
# These have to be integer multiples of the VTA tensor shape:
# :code:`BATCH`, :code:`BLOCK_IN`, and :code:`BLOCK_OUT` respectively.
#
# We've added extra operators to the matrix multiplication that apply
# shifting and clipping to the output in order to mimic a fixed-point
# matrix multiplication followed by a rectified linear activation.
# We describe the TVM dataflow graph of the fully connected layer below:
#
# .. image:: https://raw.githubusercontent.com/uwsampl/web-data/main/vta/tutorial/fc_dataflow.png
#      :align: center
#
# This computation is intentionally too large to fit onto VTA's on-chip
# buffers all at once. Therefore in the scheduling phase we'll
# rely on computation blocking strategies to break the computation down into
# manageable chunks.

# Fully connected layer dimensions: 1024 x 1024
batch_size = 1
in_channels = 1024
out_channels = 1024
assert batch_size % env.BATCH == 0
assert in_channels % env.BLOCK_IN == 0
assert out_channels % env.BLOCK_OUT == 0

# Let's derive the tiled input tensor shapes
data_shape = (batch_size // env.BATCH, in_channels // env.BLOCK_IN, env.BATCH, env.BLOCK_IN)
weight_shape = (
    out_channels // env.BLOCK_OUT,
    in_channels // env.BLOCK_IN,
    env.BLOCK_OUT,
    env.BLOCK_IN,
)
output_shape = (batch_size // env.BATCH, out_channels // env.BLOCK_OUT, env.BATCH, env.BLOCK_OUT)
num_ops = in_channels * out_channels * batch_size * 2

# Reduction axes
ic = te.reduce_axis((0, in_channels // env.BLOCK_IN), name="ic")
ic_tns = te.reduce_axis((0, env.BLOCK_IN), name="ic_tns")

# Input placeholder tensors
data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
weight = te.placeholder(weight_shape, name="weight", dtype=env.wgt_dtype)

# Copy buffers
data_buf = te.compute(data_shape, lambda *i: data(*i), "data_buf")
weight_buf = te.compute(weight_shape, lambda *i: weight(*i), "weight_buf")

# Declare matrix multiply computation
res_gemm = te.compute(
    output_shape,
    lambda bo, co, bi, ci: te.sum(
        data_buf[bo, ic, bi, ic_tns].astype(env.acc_dtype)
        * weight_buf[co, ic, ci, ic_tns].astype(env.acc_dtype),
        axis=[ic, ic_tns],
    ),
    name="res_gem",
)

# Add shift stage for fix-point normalization
res_shr = te.compute(output_shape, lambda *i: res_gemm(*i) >> env.INP_WIDTH, name="res_shr")

# Apply clipping between (0, input max value)
inp_max = (1 << (env.INP_WIDTH - 1)) - 1
res_max = te.compute(output_shape, lambda *i: tvm.te.max(res_shr(*i), 0), "res_max")
res_min = te.compute(output_shape, lambda *i: tvm.te.min(res_max(*i), inp_max), "res_min")

# Apply typecast to input data type before sending results back
res = te.compute(output_shape, lambda *i: res_min(*i).astype(env.inp_dtype), name="res")

######################################################################
# Scheduling the Computation
# --------------------------
# We'll look at a set of schedule transformations necessary to map the
# matrix multiplications onto VTA in an efficient fashion.
# Those include:
#
# - Computation blocking
# - Lowering to VTA hardware intrinsics


# Create TVM schedule
s = te.create_schedule(res.op)
# Let's look at the default TVM schedule
print(tvm.lower(s, [data, weight, res], simple_mode=True))

######################################################################
# Blocking the Computation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# The matrix multiplication is by default too large for activations or weights
# to fit on VTA's on-chip buffers all at once.
# We block the (1, 1024) by (1024, 1024) matrix multiplication into
# smaller (1, 256) by (256, 256) matrix multiplications so the intermediate
# tensors can fit on the accelerator's on-chip SRAM.
# This approach is similar to blocking techniques applied to CPUs and GPUs in
# order to increase cache hit rate.
#
# We perform blocking along each axes (the batch axis being untouched since
# we are performing singe-batch inference).
# We also leave the inner-most tensorization axes as-is in order to allow
# TVM to pattern-match tensorization.
# We show the outcome of blocking on the computation schedule in the diagram
# below:
#
# .. image:: https://raw.githubusercontent.com/uwsampl/web-data/main/vta/tutorial/blocking.png
#      :align: center
#      :width: 480px
#
# .. note::
#
#   The code after loop splitting and reordering is equivalent to the following
#   pseudo-code. We ignore the batch axis since we are only performing single-batch
#   inference in this example:
#
#   .. code-block:: c
#
#      for (int oc_out = 0; oc_out < 4; ++oc_out) {
#        // Initialization loop
#        for (int oc_inn = 0; oc_inn < 16; ++oc_inn) {
#         for (int oc_tns = 0; oc_tns < 16; ++oc_tns) {
#          int j = (oc_out * 16 + oc_inn) * 16 + oc_tns;
#          C[0][j] = 0;
#         }
#        }
#        for (int ic_out = 0; ic_out < 4; ++ic_out) {
#         // Block loop
#         for (int oc_inn = 0; oc_inn < 16; ++oc_inn) {
#          for (int ic_inn = 0; ic_inn < 16; ++ic_inn) {
#           // Tensorization loop
#           for (int oc_tns = 0; oc_tns < 16; ++oc_tns) {
#            for (int ic_tns = 0; ic_tns < 16; ++ic_tns) {
#             int i = (ic_out * 16 + ic_inn) * 16 + ic_tns;
#             int j = (oc_out * 16 + oc_inn) * 16 + oc_tns;
#             C[0][i] = C[0][i] + A[0][i] * B[j][i];
#            }
#           }
#          }
#         }
#        }
#       }
#      }

# Let's define tiling sizes (expressed in multiples of VTA tensor shape size)
b_block = 1 // env.BATCH
i_block = 256 // env.BLOCK_IN
o_block = 256 // env.BLOCK_OUT

# Tile the output tensor along the batch and output channel dimensions
# (since by default we are doing single batch inference, the split along
#  the batch dimension has no effect)
b, oc, b_tns, oc_tns = s[res].op.axis
b_out, b_inn = s[res].split(b, b_block)
oc_out, oc_inn = s[res].split(oc, o_block)
s[res].reorder(b_out, oc_out, b_inn, oc_inn)

# Move intermediate computation into each output compute tile
s[res_gemm].compute_at(s[res], oc_out)
s[res_shr].compute_at(s[res], oc_out)
s[res_max].compute_at(s[res], oc_out)
s[res_min].compute_at(s[res], oc_out)

# Apply additional loop split along reduction axis (input channel)
b_inn, oc_inn, b_tns, oc_tns = s[res_gemm].op.axis
ic_out, ic_inn = s[res_gemm].split(ic, i_block)

# Reorder axes. We move the ic_out axis all the way out of the GEMM
# loop to block along the reduction axis
s[res_gemm].reorder(ic_out, b_inn, oc_inn, ic_inn, b_tns, oc_tns, ic_tns)

# Let's look at the current TVM schedule after blocking
print(tvm.lower(s, [data, weight, res], simple_mode=True))

######################################################################
# Lowering Copies to DMA Transfers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next we set the buffer scopes to the corresponding on-chip VTA SRAM buffers.
# We move the load loops into the matrix multiply computation loop to stage
# memory loads such that they fit in the on-chip SRAM buffers.
# Finally we annotate the load/store loop outer axes with the DMA copy pragma
# to perform bulk memory transfers on VTA.

# Set scope of SRAM buffers
s[data_buf].set_scope(env.inp_scope)
s[weight_buf].set_scope(env.wgt_scope)
s[res_gemm].set_scope(env.acc_scope)
s[res_shr].set_scope(env.acc_scope)
s[res_min].set_scope(env.acc_scope)
s[res_max].set_scope(env.acc_scope)

# Block data and weight cache reads
s[data_buf].compute_at(s[res_gemm], ic_out)
s[weight_buf].compute_at(s[res_gemm], ic_out)

# Use DMA copy pragma on DRAM->SRAM operations
s[data_buf].pragma(s[data_buf].op.axis[0], env.dma_copy)
s[weight_buf].pragma(s[weight_buf].op.axis[0], env.dma_copy)

# Use DMA copy pragma on SRAM->DRAM operation
# (this implies that these copies should be performed along b_inn,
# or result axis 2)
s[res].pragma(s[res].op.axis[2], env.dma_copy)

######################################################################
# Lowering Computation to VTA Compute Intrinsics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The last phase is to lower the computation loops down to VTA hardware
# intrinsics by mapping the matrix multiplication to tensor intrinsics,
# and mapping the shift, and clipping computation to the vector ALU.

# Apply tensorization over the batch tensor tile axis
s[res_gemm].tensorize(b_tns, env.gemm)

# Add an ALU pragma over the shift and clipping operations
s[res_shr].pragma(s[res_shr].op.axis[0], env.alu)
s[res_min].pragma(s[res_min].op.axis[0], env.alu)
s[res_max].pragma(s[res_max].op.axis[0], env.alu)

# Let's look at the final lowered TVM schedule after lowering memory
# loads/stores down to DMA copy intrinsics, and the computation down to
# VTA compute intrinsics.
print(vta.lower(s, [data, weight, res], simple_mode=True))

######################################################################
# TVM Compilation and Verification
# --------------------------------
# After specifying the schedule, we can compile it into a TVM function.
# We save the module so we can send it over RPC.
# We run the function and verify it against a numpy implementation to
# ensure correctness.

# Compile the TVM module
my_gemm = vta.build(
    s, [data, weight, res], tvm.target.Target("ext_dev", host=env.target_host), name="my_gemm"
)
temp = utils.tempdir()
my_gemm.save(temp.relpath("gemm.o"))
remote.upload(temp.relpath("gemm.o"))
f = remote.load_module("gemm.o")

# Get the remote device context
ctx = remote.ext_dev(0)

# Initialize the data and weight arrays randomly in the int range of (-128, 128]
data_np = np.random.randint(-128, 128, size=(batch_size, in_channels)).astype(data.dtype)
weight_np = np.random.randint(-128, 128, size=(out_channels, in_channels)).astype(weight.dtype)

# Apply packing to the data and weight arrays from a 2D to a 4D packed layout
data_packed = data_np.reshape(
    batch_size // env.BATCH, env.BATCH, in_channels // env.BLOCK_IN, env.BLOCK_IN
).transpose((0, 2, 1, 3))
weight_packed = weight_np.reshape(
    out_channels // env.BLOCK_OUT, env.BLOCK_OUT, in_channels // env.BLOCK_IN, env.BLOCK_IN
).transpose((0, 2, 1, 3))

# Format the input/output arrays with tvm.nd.array to the DLPack standard
data_nd = tvm.nd.array(data_packed, ctx)
weight_nd = tvm.nd.array(weight_packed, ctx)
res_nd = tvm.nd.array(np.zeros(output_shape).astype(res.dtype), ctx)

# Clear stats
if env.TARGET in ["sim", "tsim"]:
    simulator.clear_stats()

# Invoke the module to perform the computation
f(data_nd, weight_nd, res_nd)

# Verify against numpy implementation
res_ref = np.dot(data_np.astype(env.acc_dtype), weight_np.T.astype(env.acc_dtype))
res_ref = res_ref >> env.INP_WIDTH
res_ref = np.clip(res_ref, 0, inp_max)
res_ref = res_ref.astype(res.dtype)
res_ref = res_ref.reshape(
    batch_size // env.BATCH, env.BATCH, out_channels // env.BLOCK_OUT, env.BLOCK_OUT
).transpose((0, 2, 1, 3))
np.testing.assert_equal(res_ref, res_nd.numpy())

# Print stats
if env.TARGET in ["sim", "tsim"]:
    sim_stats = simulator.stats()
    print("Execution statistics:")
    for k, v in sim_stats.items():
        print("\t{:<16}: {:>16}".format(k, v))

print("Successful blocked matrix multiply test!")

######################################################################
# Summary
# -------
# This tutorial demonstrates how TVM scheduling primitives can achieve
# computation blocking for a matrix multiplication example.
# This allows us to map arbitrarily large computation onto limited
# hardware accelerator resources.
#

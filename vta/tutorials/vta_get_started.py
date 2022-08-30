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
.. _vta-get-started:

Get Started with VTA
====================
**Author**: `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

This is an introduction tutorial on how to use TVM to program the VTA design.

In this tutorial, we will demonstrate the basic TVM workflow to implement
a vector addition on the VTA design's vector ALU.
This process includes specific scheduling transformations necessary to lower
computation down to low-level accelerator operations.

To begin, we need to import TVM which is our deep learning optimizing compiler.
We also need to import the VTA python package which contains VTA specific
extensions for TVM to target the VTA design.
"""
from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import vta
import numpy as np

######################################################################
# Loading in VTA Parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# VTA is a modular and customizable design. Consequently, the user
# is free to modify high-level hardware parameters that affect
# the hardware design layout.
# These parameters are specified in the :code:`vta_config.json` file by their
# :code:`log2` values.
# These VTA parameters can be loaded with the :code:`vta.get_env`
# function.
#
# Finally, the TVM target is also specified in the :code:`vta_config.json` file.
# When set to *sim*, execution will take place inside of a behavioral
# VTA simulator.
# If you want to run this tutorial on the Pynq FPGA development platform,
# follow the *VTA Pynq-Based Testing Setup* guide.

env = vta.get_env()

######################################################################
# FPGA Programming
# ----------------
# When targeting the Pynq FPGA development board, we need to configure
# the board with a VTA bitstream.

# We'll need the TVM RPC module and the VTA simulator module
from tvm import rpc
from tvm.contrib import utils
from vta.testing import simulator

# We read the Pynq RPC host IP address and port number from the OS environment
host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
port = int(os.environ.get("VTA_RPC_PORT", "9091"))

# We configure both the bitstream and the runtime system on the Pynq
# to match the VTA configuration specified by the vta_config.json file.
if env.TARGET == "pynq" or env.TARGET == "de10nano":

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
elif env.TARGET in ("sim", "tsim", "intelfocl"):
    remote = rpc.LocalSession()

    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

######################################################################
# Computation Declaration
# -----------------------
# As a first step, we need to describe our computation.
# TVM adopts tensor semantics, with each intermediate result
# represented as multi-dimensional array. The user needs to describe
# the computation rule that generates the output tensors.
#
# In this example we describe a vector addition, which requires multiple
# computation stages, as shown in the dataflow diagram below.
# First we describe the input tensors :code:`A` and :code:`B` that are living
# in main memory.
# Second, we need to declare intermediate tensors :code:`A_buf` and
# :code:`B_buf`, which will live in VTA's on-chip buffers.
# Having this extra computational stage allows us to explicitly
# stage cached reads and writes.
# Third, we describe the vector addition computation which will
# add :code:`A_buf` to :code:`B_buf` to produce :code:`C_buf`.
# The last operation is a cast and copy back to DRAM, into results tensor
# :code:`C`.
#
# .. image:: https://raw.githubusercontent.com/uwsampl/web-data/main/vta/tutorial/vadd_dataflow.png
#      :align: center

######################################################################
# Input Placeholders
# ~~~~~~~~~~~~~~~~~~
# We describe the placeholder tensors :code:`A`, and :code:`B` in a tiled data
# format to match the data layout requirements imposed by the VTA vector ALU.
#
# For VTA's general purpose operations such as vector adds, the tile size is
# :code:`(env.BATCH, env.BLOCK_OUT)`.
# The dimensions are specified in
# the :code:`vta_config.json` configuration file and are set by default to
# a (1, 16) vector.
#
# In addition, A and B's data types also needs to match the :code:`env.acc_dtype`
# which is set by the :code:`vta_config.json` file to be a 32-bit integer.

# Output channel factor m - total 64 x 16 = 1024 output channels
m = 64
# Batch factor o - total 1 x 1 = 1
o = 1
# A placeholder tensor in tiled data format
A = te.placeholder((o, m, env.BATCH, env.BLOCK_OUT), name="A", dtype=env.acc_dtype)
# B placeholder tensor in tiled data format
B = te.placeholder((o, m, env.BATCH, env.BLOCK_OUT), name="B", dtype=env.acc_dtype)

######################################################################
# Copy Buffers
# ~~~~~~~~~~~~
# One specificity of hardware accelerators, is that on-chip memory has to be
# explicitly managed.
# This means that we'll need to describe intermediate tensors :code:`A_buf`
# and :code:`B_buf` that can have a different memory scope than the original
# placeholder tensors :code:`A` and :code:`B`.
#
# Later in the scheduling phase, we can tell the compiler that :code:`A_buf`
# and :code:`B_buf` will live in the VTA's on-chip buffers (SRAM), while
# :code:`A` and :code:`B` will live in main memory (DRAM).
# We describe A_buf and B_buf as the result of a compute
# operation that is the identity function.
# This can later be interpreted by the compiler as a cached read operation.

# A copy buffer
A_buf = te.compute((o, m, env.BATCH, env.BLOCK_OUT), lambda *i: A(*i), "A_buf")
# B copy buffer
B_buf = te.compute((o, m, env.BATCH, env.BLOCK_OUT), lambda *i: B(*i), "B_buf")

######################################################################
# Vector Addition
# ~~~~~~~~~~~~~~~
# Now we're ready to describe the vector addition result tensor :code:`C`,
# with another compute operation.
# The compute function takes the shape of the tensor, as well as a lambda
# function that describes the computation rule for each position of the tensor.
#
# No computation happens during this phase, as we are only declaring how
# the computation should be done.

# Describe the in-VTA vector addition
C_buf = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT),
    lambda *i: A_buf(*i).astype(env.acc_dtype) + B_buf(*i).astype(env.acc_dtype),
    name="C_buf",
)

######################################################################
# Casting the Results
# ~~~~~~~~~~~~~~~~~~~
# After the computation is done, we'll need to send the results computed by VTA
# back to main memory.

######################################################################
# .. note::
#
#   **Memory Store Restrictions**
#
#   One specificity of VTA is that it only supports DRAM stores in the narrow
#   :code:`env.inp_dtype` data type format.
#   This lets us reduce the data footprint for memory transfers (more on this
#   in the basic matrix multiply example).
#
# We perform one last typecast operation to the narrow
# input activation data format.

# Cast to output type, and send to main memory
C = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT), lambda *i: C_buf(*i).astype(env.inp_dtype), name="C"
)

######################################################################
# This concludes the computation declaration part of this tutorial.


######################################################################
# Scheduling the Computation
# --------------------------
# While the above lines describes the computation rule, we can obtain
# :code:`C` in many ways.
# TVM asks the user to provide an implementation of the computation called
# *schedule*.
#
# A schedule is a set of transformations to an original computation that
# transforms the implementation of the computation without affecting
# correctness.
# This simple VTA programming tutorial aims to demonstrate basic schedule
# transformations that will map the original schedule down to VTA hardware
# primitives.


######################################################################
# Default Schedule
# ~~~~~~~~~~~~~~~~
# After we construct the schedule, by default the schedule computes
# :code:`C` in the following way:

# Let's take a look at the generated schedule
s = te.create_schedule(C.op)

print(tvm.lower(s, [A, B, C], simple_mode=True))

######################################################################
# Although this schedule makes sense, it won't compile to VTA.
# In order to obtain correct code generation, we need to apply scheduling
# primitives and code annotation that will transform the schedule into
# one that can be directly lowered onto VTA hardware intrinsics.
# Those include:
#
#  - DMA copy operations which will take globally-scoped tensors and copy
#    those into locally-scoped tensors.
#  - Vector ALU operations that will perform the vector add.

######################################################################
# Buffer Scopes
# ~~~~~~~~~~~~~
# First, we set the scope of the copy buffers to indicate to TVM that these
# intermediate tensors will be stored in the VTA's on-chip SRAM buffers.
# Below, we tell TVM that :code:`A_buf`, :code:`B_buf`, :code:`C_buf`
# will live in VTA's on-chip *accumulator buffer* which serves as
# VTA's general purpose register file.
#
# Set the intermediate tensors' scope to VTA's on-chip accumulator buffer
s[A_buf].set_scope(env.acc_scope)
s[B_buf].set_scope(env.acc_scope)
s[C_buf].set_scope(env.acc_scope)

######################################################################
# DMA Transfers
# ~~~~~~~~~~~~~
# We need to schedule DMA transfers to move data living in DRAM to
# and from the VTA on-chip buffers.
# We insert :code:`dma_copy` pragmas to indicate to the compiler
# that the copy operations will be performed in bulk via DMA,
# which is common in hardware accelerators.

# Tag the buffer copies with the DMA pragma to map a copy loop to a
# DMA transfer operation
s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
s[C].pragma(s[C].op.axis[0], env.dma_copy)

######################################################################
# ALU Operations
# ~~~~~~~~~~~~~~
# VTA has a vector ALU that can perform vector operations on tensors
# in the accumulator buffer.
# In order to tell TVM that a given operation needs to be mapped to the
# VTA's vector ALU, we need to explicitly tag the vector addition loop
# with an :code:`env.alu` pragma.

# Tell TVM that the computation needs to be performed
# on VTA's vector ALU
s[C_buf].pragma(C_buf.op.axis[0], env.alu)

# Let's take a look at the finalized schedule
print(vta.lower(s, [A, B, C], simple_mode=True))

######################################################################
# This concludes the scheduling portion of this tutorial.

######################################################################
# TVM Compilation
# ---------------
# After we have finished specifying the schedule, we can compile it
# into a TVM function. By default TVM compiles into a type-erased
# function that can be directly called from python side.
#
# In the following line, we use :code:`tvm.build` to create a function.
# The build function takes the schedule, the desired signature of the
# function(including the inputs and outputs) as well as target language
# we want to compile to.
#
my_vadd = vta.build(
    s, [A, B, C], tvm.target.Target("ext_dev", host=env.target_host), name="my_vadd"
)

######################################################################
# Saving the Module
# ~~~~~~~~~~~~~~~~~
# TVM lets us save our module into a file so it can loaded back later. This
# is called ahead-of-time compilation and allows us to save some compilation
# time.
# More importantly, this allows us to cross-compile the executable on our
# development machine and send it over to the Pynq FPGA board over RPC for
# execution.

# Write the compiled module into an object file.
temp = utils.tempdir()
my_vadd.save(temp.relpath("vadd.o"))

# Send the executable over RPC
remote.upload(temp.relpath("vadd.o"))

######################################################################
# Loading the Module
# ~~~~~~~~~~~~~~~~~~
# We can load the compiled module from the file system to run the code.

f = remote.load_module("vadd.o")

######################################################################
# Running the Function
# --------------------
# The compiled TVM function uses a concise C API and can be invoked from
# any language.
#
# TVM provides an array API in python to aid quick testing and prototyping.
# The array API is based on `DLPack <https://github.com/dmlc/dlpack>`_ standard.
#
# - We first create a remote context (for remote execution on the Pynq).
# - Then :code:`tvm.nd.array` formats the data accordingly.
# - :code:`f()` runs the actual computation.
# - :code:`numpy()` copies the result array back in a format that can be
#   interpreted.
#

# Get the remote device context
ctx = remote.ext_dev(0)

# Initialize the A and B arrays randomly in the int range of (-128, 128]
A_orig = np.random.randint(-128, 128, size=(o * env.BATCH, m * env.BLOCK_OUT)).astype(A.dtype)
B_orig = np.random.randint(-128, 128, size=(o * env.BATCH, m * env.BLOCK_OUT)).astype(B.dtype)

# Apply packing to the A and B arrays from a 2D to a 4D packed layout
A_packed = A_orig.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))
B_packed = B_orig.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))

# Format the input/output arrays with tvm.nd.array to the DLPack standard
A_nd = tvm.nd.array(A_packed, ctx)
B_nd = tvm.nd.array(B_packed, ctx)
C_nd = tvm.nd.array(np.zeros((o, m, env.BATCH, env.BLOCK_OUT)).astype(C.dtype), ctx)

# Invoke the module to perform the computation
f(A_nd, B_nd, C_nd)

######################################################################
# Verifying Correctness
# ---------------------
# Compute the reference result with numpy and assert that the output of the
# matrix multiplication indeed is correct

# Compute reference result with numpy
C_ref = (A_orig.astype(env.acc_dtype) + B_orig.astype(env.acc_dtype)).astype(C.dtype)
C_ref = C_ref.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))
np.testing.assert_equal(C_ref, C_nd.numpy())
print("Successful vector add test!")

######################################################################
# Summary
# -------
# This tutorial provides a walk-through of TVM for programming the
# deep learning accelerator VTA with a simple vector addition example.
# The general workflow includes:
#
# - Programming the FPGA with the VTA bitstream over RPC.
# - Describing the vector add computation via a series of computations.
# - Describing how we want to perform the computation using schedule primitives.
# - Compiling the function to the VTA target.
# - Running the compiled module and verifying it against a numpy implementation.
#
# You are more than welcome to check other examples out and tutorials
# to learn more about the supported operations, schedule primitives
# and other features supported by TVM to program VTA.
#

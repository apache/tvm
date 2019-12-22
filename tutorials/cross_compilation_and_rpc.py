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
.. _tutorial-cross-compilation-and-rpc:

Cross Compilation and RPC
=========================
**Author**: `Ziheng Jiang <https://github.com/ZihengJiang/>`_, `Lianmin Zheng <https://github.com/merrymercy/>`_

This tutorial introduces cross compilation and remote device
execution with RPC in TVM.

With cross compilation and RPC, you can **compile a program on your
local machine then run it on the remote device**. It is useful when
the remote device resource are limited, like Raspberry Pi and mobile
platforms. In this tutorial, we will use the Raspberry Pi for a CPU example
and the Firefly-RK3399 for an OpenCL example.
"""

######################################################################
# Build TVM Runtime on Device
# ---------------------------
#
# The first step is to build the TVM runtime on the remote device.
#
# .. note::
#
#   All instructions in both this section and the next section should be
#   executed on the target device, e.g. Raspberry Pi.  We assume the target
#   is running Linux.
#
# Since we do compilation on the local machine, the remote device is only used
# for running the generated code. We only need to build the TVM runtime on
# the remote device.
#
# .. code-block:: bash
#
#   git clone --recursive https://github.com/apache/incubator-tvm tvm
#   cd tvm
#   make runtime -j2
#
# After building the runtime successfully, we need to set environment variables
# in :code:`~/.bashrc` file. We can edit :code:`~/.bashrc`
# using :code:`vi ~/.bashrc` and add the line below (Assuming your TVM
# directory is in :code:`~/tvm`):
#
# .. code-block:: bash
#
#   export PYTHONPATH=$PYTHONPATH:~/tvm/python
#
# To update the environment variables, execute :code:`source ~/.bashrc`.

######################################################################
# Set Up RPC Server on Device
# ---------------------------
# To start an RPC server, run the following command on your remote device
# (Which is Raspberry Pi in this example).
#
#   .. code-block:: bash
#
#     python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
#
# If you see the line below, it means the RPC server started
# successfully on your device.
#
#    .. code-block:: bash
#
#      INFO:root:RPCServer: bind to 0.0.0.0:9090
#

######################################################################
# Declare and Cross Compile Kernel on Local Machine
# -------------------------------------------------
#
# .. note::
#
#   Now we go back to the local machine, which has a full TVM installed
#   (with LLVM).
#
# Here we will declare a simple kernel on the local machine:

import numpy as np

import tvm
from tvm import rpc
from tvm.contrib import util

n = tvm.convert(1024)
A = tvm.placeholder((n,), name='A')
B = tvm.compute((n,), lambda i: A[i] + 1.0, name='B')
s = tvm.create_schedule(B.op)

######################################################################
# Then we cross compile the kernel.
# The target should be 'llvm -target=armv7l-linux-gnueabihf' for
# Raspberry Pi 3B, but we use 'llvm' here to make this tutorial runnable
# on our webpage building server. See the detailed note in the following block.

local_demo = True

if local_demo:
    target = 'llvm'
else:
    target = 'llvm -target=armv7l-linux-gnueabihf'

func = tvm.build(s, [A, B], target=target, name='add_one')
# save the lib at a local temp folder
temp = util.tempdir()
path = temp.relpath('lib.tar')
func.export_library(path)

######################################################################
# .. note::
#
#   To run this tutorial with a real remote device, change :code:`local_demo`
#   to False and replace :code:`target` in :code:`build` with the appropriate
#   target triple for your device. The target triple which might be
#   different for different devices. For example, it is
#   :code:`'llvm -target=armv7l-linux-gnueabihf'` for Raspberry Pi 3B and
#   :code:`'llvm -target=aarch64-linux-gnu'` for RK3399.
#
#   Usually, you can query the target by running :code:`gcc -v` on your
#   device, and looking for the line starting with :code:`Target:`
#   (Though it may still be a loose configuration.)
#
#   Besides :code:`-target`, you can also set other compilation options
#   like:
#
#   * -mcpu=<cpuname>
#       Specify a specific chip in the current architecture to generate code for. By default this is inferred from the target triple and autodetected to the current architecture.
#   * -mattr=a1,+a2,-a3,...
#       Override or control specific attributes of the target, such as whether SIMD operations are enabled or not. The default set of attributes is set by the current CPU.
#       To get the list of available attributes, you can do:
#
#       .. code-block:: bash
#
#         llc -mtriple=<your device target triple> -mattr=help
#
#   These options are consistent with `llc <http://llvm.org/docs/CommandGuide/llc.html>`_.
#   It is recommended to set target triple and feature set to contain specific
#   feature available, so we can take full advantage of the features of the
#   board.
#   You can find more details about cross compilation attributes from
#   `LLVM guide of cross compilation <https://clang.llvm.org/docs/CrossCompilation.html>`_.

######################################################################
# Run CPU Kernel Remotely by RPC
# ------------------------------
# We show how to run the generated CPU kernel on the remote device.
# First we obtain an RPC session from remote device.

if local_demo:
    remote = rpc.LocalSession()
else:
    # The following is my environment, change this to the IP address of your target device
    host = '10.77.1.162'
    port = 9090
    remote = rpc.connect(host, port)

######################################################################
# Upload the lib to the remote device, then invoke a device local
# compiler to relink them. Now `func` is a remote module object.

remote.upload(path)
func = remote.load_module('lib.tar')

# create arrays on the remote device
ctx = remote.cpu()
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
# the function will run on the remote device
func(a, b)
np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

######################################################################
# When you want to evaluate the performance of the kernel on the remote
# device, it is important to avoid the overhead of network.
# :code:`time_evaluator` will returns a remote function that runs the
# function over number times, measures the cost per run on the remote
# device and returns the measured cost. Network overhead is excluded.

time_f = func.time_evaluator(func.entry_name, ctx, number=10)
cost = time_f(a, b).mean
print('%g secs/op' % cost)

#########################################################################
# Run OpenCL Kernel Remotely by RPC
# ---------------------------------
# For remote OpenCL devices, the workflow is almost the same as above.
# You can define the kernel, upload files, and run via RPC.
#
# .. note::
#
#    Raspberry Pi does not support OpenCL, the following code is tested on
#    Firefly-RK3399. You may follow this `tutorial <https://gist.github.com/mli/585aed2cec0b5178b1a510f9f236afa2>`_
#    to setup the OS and OpenCL driver for RK3399.
#
#    Also we need to build the runtime with OpenCL enabled on rk3399 board. In the TVM
#    root directory, execute
#
# .. code-block:: bash
#
#    cp cmake/config.cmake .
#    sed -i "s/USE_OPENCL OFF/USE_OPENCL ON/" config.cmake
#    make runtime -j4
#
# The following function shows how we run an OpenCL kernel remotely

def run_opencl():
    # NOTE: This is the setting for my rk3399 board. You need to modify
    # them according to your environment.
    target_host = "llvm -target=aarch64-linux-gnu"
    opencl_device_host = '10.77.1.145'
    opencl_device_port = 9090

    # create schedule for the above "add one" compute declaration
    s = tvm.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=32)
    s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
    func = tvm.build(s, [A, B], "opencl", target_host=target_host)

    remote = rpc.connect(opencl_device_host, opencl_device_port)

    # export and upload
    path = temp.relpath('lib_cl.tar')
    func.export_library(path)
    remote.upload(path)
    func = remote.load_module('lib_cl.tar')

    # run
    ctx = remote.cl()
    a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
    b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
    func(a, b)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)
    print("OpenCL test passed!")

######################################################################
# Summary
# -------
# This tutorial provides a walk through of cross compilation and RPC
# features in TVM.
#
# - Set up an RPC server on the remote device.
# - Set up the target device configuration to cross compile the kernels on the
#   local machine.
# - Upload and run the kernels remotely via the RPC API.

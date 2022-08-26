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
.. _tutorial-deploy-model-on-D1:

Deploy the test Model on D1
===========================

This is an example of using Relay to deploy mobilenet on Allwinner D1.
"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

import tvm
import tvm.relay as relay
from tvm import rpc
import tvm.relay.testing
from tvm.contrib import utils, graph_executor as runtime

# from apps.benchmark.util import get_network
from tvm.relay.op.contrib import csinn

import numpy as np


######################################################################
# .. _build-tvm-runtime-on-host:
#
# Cross compile TVM Runtime on Host
# ---------------------------------
#
# The first step is to build CSI-NN2 Compute Library and cross-compile
# the TVM runtime on the Host.
#
# .. note::
#
#   There is no toolchain in D1 development board, so we need to
#   cross-compile runtime and tvm_rpc in local machine then copy files
#   to board.
#
# Clone CSINN2 source code and download toolchain
# Build x86 and c906 lib
#
# .. code-block:: bash
#
#   git clone https://github.com/T-head-Semi/csi-nn2.git
#   cd csi-nn2
#   ./script/download_toolchain
#   make nn2_ref_x86;
#   make nn2_c906;
#   cd x86_build; make install; cd -
#   cd riscv_build; make install; cd -

# Build TVM with CSINN2 support for host
# Edit config.cmake
#   set(USE_OPENMP gnu)
#   set(USE_CSINN /path/to/csi-nn2)
#   set(USE_CSINN_DEVICE_RUNTIME X86)
#
# .. code-block:: bash
#
#   cd tvm
#   mkdir build
#   cp cmake/config.cmake build
#   cd build
#   cmake ..
#   make -j4

# Build TVM runtime and rpc support for D1
# Edit config.cmake
#   set(USE_CPP_RPC ON)
#   set(USE_LIBBACKTRACE OFF)
#   set(USE_CSINN /path/to/csi-nn2)
#   set(USE_CSINN_DEVICE_RUNTIME C906)
#
# .. code-block:: bash
#
#   cd tvm
#   mkdir build-rv
#   cp cmake/config.cmake build-rv
#   cd build-rv
#   cmake ..
#   make runtime -j4 tvm_rpc
#
# After building runtime successfully, we need to copy tvm_rpc and libs
# which used on D1
#

######################################################################
# Set Up RPC Server on Device
# ---------------------------
# To start an RPC server, run the following command on your remote device
#
#   .. code-block:: bash
#
#     ./tvm_rpc server --host=172.16.202.11 --port=9090
#
# If you see the line below, it means the RPC server started
# successfully on your device.
#
#    .. code-block:: bash
#
#      rpc_server.cc:130: bind to 172.16.202.11:9090
#

#################################################################
# Define a Network
# ----------------
# First, we need to define the network with relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX, PyTorch, and TensorFlow
# (see :ref:`front end tutorials<tutorial-frontend>`).
#
# CSI-NN2 only support with NCHW layout.
# You can use :ref:`ConvertLayout <convert-layout-usage>` pass to do the layout conversion in TVM.


def get_network(name, batch_size, layout="NCHW", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, random_params=True)

    return mod, params, input_shape, output_shape


######################################################################
# Compile The Graph
# -----------------
# To compile the graph, we call the :py:func:`relay.build` function
# with the graph configuration and parameters. However, You cannot to
# deploy a x86 program on a device with RISC-V instruction set. It means
# Relay also needs to know the compilation option of target device,
# apart from arguments :code:`net` and :code:`params` to specify the
# deep learning workload. Actually, the option matters, different option
# will lead to very different performance.

######################################################################
# If we run the example on our x86 server for demonstration, we can simply
# set it as :code:`llvm`. If running it on the D1, we need to specify its
# instruction set. Set :code:`local_demo` to False if you want to run
# this tutorial with a real device or QEMU.

local_demo = True

if local_demo:
    target = "llvm"
else:
    target = "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c"


network = "mobilenet"
use_sparse = False
batch_size = 1
layout = "NCHW"
dtype = "float32"
print("Get model...")
mod, params, input_shape, output_shape = get_network(
    network, batch_size, layout, dtype=dtype, use_sparse=use_sparse
)

with tvm.transform.PassContext(opt_level=3):
    mod = csinn.partition_for_csinn(mod, params)
    lib = relay.build(mod, target, params=params)

# After `relay.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.so")
if local_demo:
    lib.export_library(lib_fname)
else:
    lib.export_library(lib_fname, cc="riscv64-unknown-linux-gnu-g++")

######################################################################
# Deploy the Model Remotely by RPC
# --------------------------------
# With RPC, you can deploy the model remotely from your host machine
# to the remote device.

# obtain an RPC session from remote device.
if local_demo:
    remote = rpc.LocalSession()
else:
    # The following is my environment, change this to the IP address of your target device
    host = "127.0.0.1"
    port = 9090
    remote = rpc.connect(host, port)

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module("net.so")

# create the remote runtime module
dev = remote.cpu(0)
module = runtime.GraphModule(rlib["default"](dev))
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=1, number=1, min_repeat_ms=500))

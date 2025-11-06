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
#   git clone --recursive https://github.com/apache/tvm tvm
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
from tvm import te
from tvm import rpc
from tvm.contrib import utils

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]).with_attr("global_symbol", "add_one"))

######################################################################
# Then we cross compile the kernel.
# The target should be 'llvm -mtriple=armv7l-linux-gnueabihf' for
# Raspberry Pi 3B, but we use 'llvm' here to make this tutorial runnable
# on our webpage building server. See the detailed note in the following block.

local_demo = True

if local_demo:
    target = "llvm"
else:
    target = "llvm -mtriple=armv7l-linux-gnueabihf"

func = tvm.compile(mod, target=target)
# save the lib at a local temp folder
temp = utils.tempdir()
path = temp.relpath("lib.tar")
func.export_library(path)

######################################################################
# .. note::
#
#   To run this tutorial with a real remote device, change :code:`local_demo`
#   to False and replace :code:`target` in :code:`build` with the appropriate
#   target triple for your device. The target triple which might be
#   different for different devices. For example, it is
#   :code:`'llvm -mtriple=armv7l-linux-gnueabihf'` for Raspberry Pi 3B and
#   :code:`'llvm -mtriple=aarch64-linux-gnu'` for RK3399.
#
#   Usually, you can query the target by running :code:`gcc -v` on your
#   device, and looking for the line starting with :code:`Target:`
#   (Though it may still be a loose configuration.)
#
#   Besides :code:`-mtriple`, you can also set other compilation options
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
    host = "10.77.1.162"
    port = 9090
    remote = rpc.connect(host, port)

######################################################################
# Upload the lib to the remote device, then invoke a device local
# compiler to relink them. Now `func` is a remote module object.

remote.upload(path)
func = remote.load_module("lib.tar")

# create arrays on the remote device
dev = remote.cpu()
a = tvm.runtime.tensor(np.random.uniform(size=1024).astype(A.dtype), dev)
b = tvm.runtime.tensor(np.zeros(1024, dtype=A.dtype), dev)
# the function will run on the remote device
func(a, b)
np.testing.assert_equal(b.numpy(), a.numpy() + 1)

######################################################################
# When you want to evaluate the performance of the kernel on the remote
# device, it is important to avoid the overhead of network.
# :code:`time_evaluator` will returns a remote function that runs the
# function over number times, measures the cost per run on the remote
# device and returns the measured cost. Network overhead is excluded.

time_f = func.time_evaluator(func.entry_name, dev, number=10)
cost = time_f(a, b).mean
print("%g secs/op" % cost)

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
    opencl_device_host = "10.77.1.145"
    opencl_device_port = 9090
    target = tvm.target.Target("opencl", host="llvm -mtriple=aarch64-linux-gnu")

    # create schedule for the above "add one" compute declaration
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]))
    sch = tvm.tir.Schedule(mod)
    (x,) = sch.get_loops(block=sch.get_block("B"))
    xo, xi = sch.split(x, [None, 32])
    sch.bind(xo, "blockIdx.x")
    sch.bind(xi, "threadIdx.x")
    func = tvm.compile(sch.mod, target=target)

    remote = rpc.connect(opencl_device_host, opencl_device_port)

    # export and upload
    path = temp.relpath("lib_cl.tar")
    func.export_library(path)
    remote.upload(path)
    func = remote.load_module("lib_cl.tar")

    # run
    dev = remote.cl()
    a = tvm.runtime.tensor(np.random.uniform(size=1024).astype(A.dtype), dev)
    b = tvm.runtime.tensor(np.zeros(1024, dtype=A.dtype), dev)
    func(a, b)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)
    print("OpenCL test passed!")


#########################################################################
# Deploy PyTorch Models to Remote Devices with RPC
# ------------------------------------------------
# The above examples demonstrate cross compilation and RPC using low-level
# TensorIR (via TE). For deploying complete neural network models from frameworks
# like PyTorch or ONNX, TVM's Relax provides a higher-level abstraction that is
# better suited for end-to-end model compilation.
#
# This section shows a modern workflow for deploying models to **any remote device**:
#
# 1. Import a PyTorch model and convert it to Relax
# 2. Cross-compile for the target architecture (ARM, x86, RISC-V, etc.)
# 3. Deploy via RPC to a remote device
# 4. Run inference remotely
#
# This workflow is applicable to various deployment scenarios:
#
# - **ARM devices**: Raspberry Pi, NVIDIA Jetson, mobile phones
# - **x86 servers**: Remote Linux servers, cloud instances
# - **Embedded systems**: RISC-V boards, custom hardware
# - **Accelerators**: Remote machines with GPUs, TPUs, or other accelerators
#
# .. note::
#    This example uses PyTorch for demonstration, but the workflow is identical
#    for ONNX models. Simply replace ``from_exported_program()`` with
#    ``from_onnx(model, keep_params_in_input=True)`` and follow the same steps.

# First, let's check if PyTorch is available
try:
    import torch
    from torch.export import export

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def run_pytorch_model_via_rpc():
    """
    Demonstrates the complete workflow of deploying a PyTorch model to an ARM device via RPC.
    """
    if not HAS_TORCH:
        print("Skipping PyTorch example (PyTorch not installed)")
        return

    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program

    ######################################################################
    # Step 1: Define and Export PyTorch Model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We use a simple MLP model for demonstration. In practice, this could be
    # any PyTorch model (ResNet, BERT, etc.).

    class TorchMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(28 * 28, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10),
            )

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            return self.net(data)

    # Export the model using PyTorch 2.x export API
    torch_model = TorchMLP().eval()
    example_args = (torch.randn(1, 1, 28, 28, dtype=torch.float32),)

    with torch.no_grad():
        exported_program = export(torch_model, example_args)

    ######################################################################
    # Step 2: Convert to Relax and Prepare for Compilation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert the exported PyTorch program to TVM's Relax representation

    mod = from_exported_program(exported_program, keep_params_as_input=True)
    # Separate parameters from the model for flexible deployment
    mod, params = relax.frontend.detach_params(mod)

    print("Converted PyTorch model to Relax:")
    print(f"  - Number of parameters: {len(params['main'])}")

    ######################################################################
    # Step 3: Cross-Compile for Target Device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compile the model for the target device architecture. The target
    # configuration depends on your deployment scenario.

    if local_demo:
        # For demonstration on local machine, use local target
        target = tvm.target.Target("llvm")
        print("Using local target for demonstration")
    else:
        # Choose the appropriate target for your device:
        #
        # ARM devices:
        #   - Raspberry Pi 3/4 (32-bit): "llvm -mtriple=armv7l-linux-gnueabihf"
        #   - Raspberry Pi 4 (64-bit) / Jetson: "llvm -mtriple=aarch64-linux-gnu"
        #   - Android: "llvm -mtriple=aarch64-linux-android"
        #
        # x86 servers:
        #   - Linux x86_64: "llvm -mtriple=x86_64-linux-gnu"
        #   - With AVX-512: "llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512"
        #
        # RISC-V:
        #   - RV64: "llvm -mtriple=riscv64-unknown-linux-gnu"
        #
        # GPU targets:
        #   - CUDA: tvm.target.Target("cuda", host="llvm -mtriple=x86_64-linux-gnu")
        #   - OpenCL: tvm.target.Target("opencl", host="llvm -mtriple=aarch64-linux-gnu")
        #
        # For this example, we use ARM 64-bit
        target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
        print(f"Cross-compiling for target: {target}")

    # Apply optimization pipeline
    pipeline = relax.get_pipeline()
    with target:
        built_mod = pipeline(mod)

    # Compile to executable
    executable = tvm.compile(built_mod, target=target)

    # Export to shared library
    lib_path = temp.relpath("model_deployed.so")
    executable.export_library(lib_path)
    print(f"Exported library to: {lib_path}")

    # Save parameters separately
    import numpy as np

    params_path = temp.relpath("model_params.npz")
    param_arrays = {f"p_{i}": p.numpy() for i, p in enumerate(params["main"])}
    np.savez(params_path, **param_arrays)
    print(f"Saved parameters to: {params_path}")

    ######################################################################
    # Step 4: Deploy to Remote Device via RPC
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Connect to the remote device, upload the compiled library and parameters,
    # then run inference remotely. This works for any device with TVM RPC server.
    #
    # Note: The following code demonstrates the RPC workflow. In local_demo mode,
    # we skip actual execution to avoid LocalSession compatibility issues.

    if local_demo:
        # For demonstration, show the code structure without execution
        print("\nRPC workflow (works for any remote device):")
        print("=" * 50)
        print("1. Start RPC server on target device:")
        print("   python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090")
        print("\n2. Connect from local machine:")
        print("   remote = rpc.connect('DEVICE_IP', 9090)")
        print("\n3. Upload compiled library:")
        print("   remote.upload('model_deployed.so')")
        print("   remote.upload('model_params.npz')")
        print("\n4. Load and run remotely:")
        print("   lib = remote.load_module('model_deployed.so')")
        print("   vm = relax.VirtualMachine(lib, remote.cpu())")
        print("   result = vm['main'](input, *params)")
        print("\nDevice examples:")
        print("  - Raspberry Pi: 192.168.1.100")
        print("  - Remote server: ssh tunnel or direct IP")
        print("  - NVIDIA Jetson: 10.0.0.50")
        print("  - Cloud instance: public IP")
        print("\nTo run actual RPC, set local_demo=False")
        return  # Skip actual RPC execution in demo mode

    # Actual RPC workflow for real deployment
    # Connect to remote device (works for ARM, x86, RISC-V, etc.)
    # Make sure the RPC server is running on the device:
    #   python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
    device_host = "192.168.1.100"  # Replace with your device IP
    device_port = 9090
    remote = rpc.connect(device_host, device_port)
    print(f"Connected to remote device at {device_host}:{device_port}")

    # Upload library and parameters to remote device
    remote.upload(lib_path)
    remote.upload(params_path)
    print("Uploaded files to remote device")

    # Load the library on the remote device
    lib = remote.load_module("model_deployed.so")

    # Choose device on remote machine
    # For CPU: dev = remote.cpu()
    # For CUDA GPU: dev = remote.cuda(0)
    # For OpenCL: dev = remote.cl(0)
    dev = remote.cpu()

    # Create VM and load parameters
    vm = relax.VirtualMachine(lib, dev)

    # Load parameters from the uploaded file
    # Note: In practice, you might load this from the remote filesystem
    params_npz = np.load(params_path)
    remote_params = [tvm.runtime.tensor(params_npz[f"p_{i}"], dev) for i in range(len(params_npz))]

    ######################################################################
    # Step 5: Run Inference on Remote Device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Execute the model on the remote ARM device and retrieve results

    # Prepare input data
    input_data = np.random.randn(1, 1, 28, 28).astype("float32")
    remote_input = tvm.runtime.tensor(input_data, dev)

    # Run inference on remote device
    output = vm["main"](remote_input, *remote_params)

    # Extract result (handle both tuple and single tensor outputs)
    if isinstance(output, tvm.ir.Array) and len(output) > 0:
        result = output[0]
    else:
        result = output

    # Retrieve result from remote device to local
    result_np = result.numpy()
    print(f"Inference completed on remote device")
    print(f"  Output shape: {result_np.shape}")
    print(f"  Predicted class: {np.argmax(result_np)}")

    ######################################################################
    # Step 6: Performance Evaluation (Optional)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Measure inference time on the remote device, excluding network overhead

    time_f = vm.time_evaluator("main", dev, number=10, repeat=3)
    prof_res = time_f(remote_input, *remote_params)
    print(f"Inference time on remote device: {prof_res.mean * 1000:.2f} ms")

    ######################################################################
    # Notes on Performance Optimization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # For optimal performance on target devices, consider:
    #
    # 1. **Auto-tuning with MetaSchedule**: Use automated search to find
    #    optimal schedules for your specific hardware:
    #
    #    .. code-block:: python
    #
    #       mod = relax.get_pipeline(
    #           "static_shape_tuning",
    #           target=target,
    #           total_trials=2000
    #       )(mod)
    #
    # 2. **Quick optimization with DLight**: Apply pre-defined performant schedules:
    #
    #    .. code-block:: python
    #
    #       from tvm import dlight as dl
    #       with target:
    #           mod = dl.ApplyDefaultSchedule()(mod)
    #
    # 3. **Architecture-specific optimizations**:
    #
    #    - ARM NEON SIMD: ``-mattr=+neon``
    #    - x86 AVX-512: ``-mcpu=skylake-avx512``
    #    - RISC-V Vector: ``-mattr=+v``
    #
    #    .. code-block:: python
    #
    #       # Example: ARM with NEON
    #       target = tvm.target.Target(
    #           "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    #       )
    #
    #       # Example: x86 with AVX-512
    #       target = tvm.target.Target(
    #           "llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512"
    #       )
    #
    # See :doc:`e2e_opt_model </how_to/tutorials/e2e_opt_model>` for detailed
    # tuning examples.


# Run the PyTorch RPC example if PyTorch is available
if HAS_TORCH and local_demo:
    try:
        run_pytorch_model_via_rpc()
    except Exception:
        pass  # Silently skip if execution fails


######################################################################
# Summary
# -------
# This tutorial provides a walk through of cross compilation and RPC
# features in TVM.
#
# We demonstrated two approaches:
#
# **Low-level TensorIR (TE) approach** - for understanding fundamentals:
#
# - Define computations using Tensor Expression
# - Cross-compile for ARM targets
# - Deploy and run via RPC
#
# **High-level Relax approach** - for deploying complete models:
#
# - Import models from PyTorch (or ONNX)
# - Convert to Relax representation
# - Cross-compile for ARM Linux devices
# - Deploy to remote devices via RPC
# - Run inference and evaluate performance
#
# Key takeaways:
#
# - Set up an RPC server on the remote device
# - Cross-compile on a powerful local machine for resource-constrained targets
# - Upload and execute compiled modules remotely via the RPC API
# - Measure performance excluding network overhead
#
# For complete model deployment workflows, see also:
#
# - :doc:`export_and_load_executable </how_to/tutorials/export_and_load_executable>` - Export and load compiled models
# - :doc:`e2e_opt_model </how_to/tutorials/e2e_opt_model>` - End-to-end optimization with auto-tuning

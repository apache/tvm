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

import os
import tvm
import numpy as np
from tvm import relax
from tvm.contrib import utils, ndk
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno
from tvm.contrib import dlpack as dl
import tvm.testing
from tvm.rpc import connect_tracker


def get_target(backend, is_adreno=False):
    """
    Get the target for the Adreno GPU.

    Returns
    -------
    tvm.target.Target
        The target for the Adreno GPU.
    """
    target = tvm.target.adreno(backend=backend)
    if is_adreno:
        target = tvm.target.adreno(cfg="texture", backend=backend)
    return target


def get_rpc():
    """
    Establish an RPC connection to the remote device.

    Returns
    -------
    tvm.rpc.RPCSession or None
        The RPC session object if RPC_TARGET is set; otherwise, None.
    """
    rpc_target = os.getenv("RPC_TARGET", None)
    if rpc_target:
        host = os.getenv("TVM_TRACKER_HOST", "localhost")
        port = int(os.getenv("TVM_TRACKER_PORT", 9090))
        device_key = os.getenv("RPC_DEVICE_KEY", "android")
        tracker = connect_tracker(host, port)
        return tracker.request(device_key, priority=1, session_timeout=1000)
    else:
        return None


def get_unique_dso_lib():
    """
    Generate a unique shared library filename based on environment variables.

    Returns
    -------
    str
        The unique shared library filename.
    """
    rpc_tracker_port = os.getenv("TVM_TRACKER_PORT", "")
    device_port = os.getenv("DEVICE_LISTEN_PORT", "")
    return f"dev_lib_cl-{rpc_tracker_port}-{device_port}.so"


def run_cpu(mod, inputs, save_lib=False):
    """
    Run the Relax module on the local CPU for verification.

    Parameters
    ----------
    mod : tvm.IRModule
        The Relax IRModule to execute.
    inputs : list of numpy.ndarray
        The input data for the module.
    save_lib : bool, optional
        Whether to save the compiled library. Default is False.

    Returns
    -------
    tvm.runtime.NDArray or tuple of tvm.runtime.NDArray
        The output from the module execution.
    """
    print("Running on local CPU for verification")
    target = tvm.target.Target("llvm")
    ex = relax.build(mod, target)
    if save_lib:
        ex.export_library("mod.so")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(ex, dev)
    inputs = [tvm.nd.array(inp, dev) for inp in inputs]
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    return tvm_output


def build_run(mod, inputs, backend, is_adreno=False):
    remote = get_rpc()
    target = get_target(backend, is_adreno)
    if remote is None:
        tgt = tvm.target.Target(target, host="llvm")
    else:
        tgt = tvm.target.Target(target, host="llvm -mtriple=aarch64-linux-gnu")
    relax_pipeline = relax.pipeline.get_default_pipeline(tgt)
    tir_pipeline = tvm.tir.get_default_tir_pipeline(tgt)
    mod = relax_pipeline(mod)
    ex = tvm.compile(mod, tgt, tir_pipeline=tir_pipeline)

    if remote is None:
        # local execution
        if "opencl" in backend:
            dev = tvm.opencl(0)
        elif "vulkan" in backend:
            dev = tvm.vulkan(0)
        else:
            raise RuntimeError("Unsupported backend")

        if "vdevice" in mod.global_infos:
            device_arr = [dev for ii in range(len(mod.global_infos["vdevice"]))]
        else:
            device_arr = [dev]
        vm = relax.VirtualMachine(ex, device_arr)
    else:
        # remote execution
        temp = utils.tempdir()
        filename = get_unique_dso_lib()
        file_path = temp.relpath(filename)
        ex.export_library(
            file_path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"]
        )

        remote.upload(file_path)
        rexec = remote.load_module(filename)

        if "opencl" in backend:
            dev = remote.cl(0)
        elif "vulkan" in backend:
            dev = remote.vulkan(0)
        else:
            raise RuntimeError("Unsupported backend")

        if "vdevice" in mod.global_infos:
            device_arr = [dev for ii in range(len(mod.global_infos["vdevice"]))]
        else:
            device_arr = [dev]

        vm = relax.VirtualMachine(rexec, device_arr)

    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs]
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")

    if isinstance(tvm_output, tuple):
        tvm_output = (out.numpy() for out in tvm_output)
    else:
        tvm_output = tvm_output.numpy()

    if remote:
        remote.get_function("CloseRPCConnection")()
    return tvm_output


def verify(mod, backend):

    if backend not in ["opencl", "vulkan"]:
        raise ValueError(f"Unsupported API: {backend}. Must be 'opencl' or 'vulkan'.")

    inputs = []
    for arg in mod["main"].params:
        shape = tuple(shape_val.value for shape_val in arg.struct_info.shape.values)
        inputs.append(np.random.uniform(0, 1, size=shape).astype(arg.struct_info.dtype))

    ret1 = build_run(mod, inputs, backend, True)
    ret2 = build_run(mod, inputs, backend)

    if isinstance(ret1, tuple):
        for val1, val2 in zip(ret1, ret2):
            tvm.testing.assert_allclose(val1, ret2, rtol=1e-3, atol=1e-3)
    else:
        tvm.testing.assert_allclose(ret1, ret2, rtol=1e-3, atol=1e-3)

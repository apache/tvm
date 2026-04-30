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
import tempfile

import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.contrib import ndk

# Test Infra


class run_time_check:
    def __init__(self, device):
        self.device = device

    def check(self):
        # Ensure adreno specific tests
        if self.device == "real":
            return "ADRENO_TARGET" in os.environ

        # Adreno CI
        if "ADRENO_TARGET" in os.environ:
            return True

        # Tests that can run on generic targets too
        elif self.device == "opencl":
            return tvm.opencl().exist
        elif self.device == "vulkan":
            return tvm.vulkan().exist
        elif self.device == "any":
            return tvm.opencl().exist or tvm.vulkan().exist
        else:
            return False

    def __call__(self):
        return self.check


# OpenCL or Vulkan
requires_adreno_opencl_vulkan = tvm.testing.Feature(
    "adreno_opencl_vulkan",
    "Adreno Vulkan Or OpenCL",
    run_time_check=run_time_check("any")(),
    parent_features="gpu" if "ADRENO_TARGET" not in os.environ else "rpc",
)

# Any Vulkan
requires_adreno_vulkan = tvm.testing.Feature(
    "adreno_vulkan",
    "Adreno Vulkan",
    target_kind_enabled="vulkan",
    run_time_check=lambda: tvm.runtime.enabled("vulkan") and run_time_check("vulkan").check(),
    parent_features="gpu" if "ADRENO_TARGET" not in os.environ else "rpc",
)

# Any OpenCL
requires_adreno_opencl = tvm.testing.Feature(
    "adreno_opencl",
    "Adreno OpenCL",
    target_kind_enabled="opencl",
    run_time_check=lambda: tvm.runtime.enabled("opencl") and run_time_check("opencl").check(),
    parent_features="gpu" if "ADRENO_TARGET" not in os.environ else "rpc",
)

# Real Adreno GPU OpenCL Target
requires_adreno_opencl_real = tvm.testing.Feature(
    "adreno_opencl_real",
    "Adreno OpenCL Real",
    target_kind_enabled="opencl",
    run_time_check=lambda: tvm.runtime.enabled("opencl") and run_time_check("real").check(),
    parent_features="rpc",
)

# CLML Codegen
requires_adreno_clml = tvm.testing.Feature(
    "adreno_clml",
    "Adreno OpenCLML",
    run_time_check=lambda: tvm.get_global_func(
        "relax.is_openclml_runtime_enabled", allow_missing=True
    )
    is not None,
    target_kind_enabled="opencl",
    parent_features="opencl" if "ADRENO_TARGET" not in os.environ else "rpc",
)


def is_target_available(target):
    if "clml" in target.attrs.get("keys", []) and "ADRENO_TARGET" not in os.environ:
        return False
    return True


class SessionManager:
    def __init__(self):
        self.is_remote = SessionManager.is_target_rpc()

    def __enter__(self):
        if self.is_remote:
            self.RPC_TRACKER_HOST = os.getenv("TVM_TRACKER_HOST", "localhost")
            self.RPC_TRACKER_PORT = int(os.getenv("TVM_TRACKER_PORT", 7979))
            self.RPC_DEVICE_KEY = os.getenv("RPC_DEVICE_KEY", "android")

            self.tracker = tvm.rpc.connect_tracker(self.RPC_TRACKER_HOST, self.RPC_TRACKER_PORT)
            self.rpc = self.tracker.request(self.RPC_DEVICE_KEY, priority=0, session_timeout=600)
        else:
            self.rpc = tvm.rpc.LocalSession()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.rpc.get_function("CloseRPCConnection")()

    def load_module(self, ex: relax.VMExecutable):
        with tempfile.TemporaryDirectory() as tempdir:
            file_name = "vm_library.so"
            file_path = os.path.join(tempdir, file_name)
            if self.is_remote:
                ex.export_library(
                    file_path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"]
                )
            else:
                ex.export_library(file_path)

            self.rpc.upload(file_path)
            rexec = self.rpc.load_module(file_name)
        return rexec

    def device(self, device: str):
        return self.rpc.device(device)

    @staticmethod
    def is_target_rpc():
        """
        Checks if the target is a remote device.

        Returns
        -------
        bool: True if RPC_TARGET is set, False otherwise
        """
        return os.environ.get("ADRENO_TARGET") == "adreno"


def run_local(mod, inputs, target):
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
    ex = relax.build(mod, target)
    dev = tvm.cpu()
    vm = relax.VirtualMachine(ex, dev)
    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs]
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    if isinstance(tvm_output, tuple):
        tvm_output = tuple(out.numpy() for out in tvm_output)
    else:
        tvm_output = (tvm_output.numpy(),)
    return tvm_output


def build_and_run(mod, inputs, tgt):
    if SessionManager.is_target_rpc():
        tgt = tvm.target.Target(tgt, host={"kind": "llvm", "mtriple": "aarch64-linux-gnu"})
    else:
        tgt = tvm.target.Target(tgt, host={"kind": "llvm"})

    relax_pipeline = relax.pipeline.get_default_pipeline(tgt)
    tir_pipeline = tvm.tirx.get_default_tir_pipeline(tgt)
    mod = relax_pipeline(mod)

    ex = tvm.compile(mod, tgt, tir_pipeline=tir_pipeline)

    with SessionManager() as sess:
        rexec = sess.load_module(ex)
        dev = sess.device(tgt.kind.name)

        if "vdevice" in mod.global_infos:
            device_arr = [dev for ii in range(len(mod.global_infos["vdevice"]))]
        else:
            device_arr = [dev]
        vm = relax.VirtualMachine(rexec, device_arr)
        inputs = [tvm.runtime.tensor(ip, dev) for ip in inputs]
        vm.set_input("main", *inputs)

        vm.invoke_stateful("main")

        tvm_output = vm.get_outputs("main")
        if isinstance(tvm_output, tuple):
            tvm_output = tuple(out.numpy() for out in tvm_output)
        else:
            tvm_output = (tvm_output.numpy(),)

    return tvm_output


def verify_results(mod, target, ref_target):
    if not is_target_available(target):
        print("Skipping Eval Tests", flush=True)
        return

    inputs = []
    for arg in mod["main"].params:
        shape = tuple(shape_val.value for shape_val in arg.struct_info.shape.values)
        inputs.append(np.random.uniform(0, 1, size=shape).astype(arg.struct_info.dtype))

    mod_org, mod_ref = mod, mod.clone()

    mod_ref = tvm.relax.transform.DecomposeOpsForInference()(mod_ref)
    if ref_target.kind.name == "llvm":
        rs_ref = run_local(mod_ref, inputs, ref_target)
    else:
        rs_ref = build_and_run(mod_ref, inputs, ref_target)

    rs_org = build_and_run(mod_org, inputs, target)

    for vl_org, vl_ref in zip(rs_org, rs_ref):
        tvm.testing.assert_allclose(vl_org, vl_ref, rtol=1e-3, atol=1e-3)

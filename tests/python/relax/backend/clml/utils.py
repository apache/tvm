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
"""Run utils for CLML integration operator tests"""
import pytest
import numpy as np
import json
import tvm
import tvm.testing
import copy

from tvm import relax, rpc
from tvm.relax import transform
from tvm.s_tir import dlight as dl
from tvm.contrib import utils, ndk
from tvm.relax.backend.adreno.clml import OpenCLMLOffLoad

import os
from tvm import rpc as _rpc


def get_rpc():
    rpc_target = os.getenv("RPC_TARGET", None)
    if rpc_target:
        connection_type = "tracker"
        host = os.getenv("TVM_TRACKER_HOST", "localhost")
        port = int(os.getenv("TVM_TRACKER_PORT", 9090))
        target = "opencl"
        target_host = {"kind": "llvm", "mtriple": "aarch64-linux-gnu"}
        device_key = os.getenv("RPC_DEVICE_KEY", "android")
        cross_compile = os.getenv("TVM_NDK_CC", "aarch64-linux-android-g++")
        tracker = _rpc.connect_tracker(host, port)
        return tracker.request(device_key, priority=1, session_timeout=1000)
    else:
        return None


def build_and_run(
    mod,
    inputs_np,
    target,
    rpc=None,
    load_path="vm_library.so",
):
    tgt = tvm.target.Target(target, host={"kind": "llvm", "mtriple": "aarch64-linux-gnu"})
    relax_pipeline = relax.pipeline.get_default_pipeline(tgt)
    tir_pipeline = tvm.tir.get_default_tir_pipeline(tgt)

    ex = tvm.compile(mod, tgt, relax_pipeline=relax_pipeline, tir_pipeline=tir_pipeline)
    temp = utils.tempdir()
    path = temp.relpath(load_path)
    path = "./" + load_path
    ex.export_library(path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"])

    rpc.upload(path)
    rexec = rpc.load_module(load_path)
    dev = rpc.cl(0)
    vm = relax.VirtualMachine(rexec, dev)

    f = vm["main"]
    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs_np]
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main").numpy()
    if rpc:
        rpc.get_function("CloseRPCConnection")()
    return tvm_output


def run_compare(mod, inputs, params_np):
    clml_mod = copy.deepcopy(mod)
    mod = tvm.relax.transform.BindParams("main", params_np)(mod)
    clml_mod = tvm.relax.transform.BindParams("main", params_np)(clml_mod)

    rpc = get_rpc()
    if rpc is None:
        return

    ref = build_and_run(
        mod,
        inputs,
        tvm.target.Target("qcom/adreno-opencl"),
        rpc=rpc,
        load_path="vm_library_opencl.so",
    )

    rpc = get_rpc()
    out = build_and_run(
        clml_mod,
        inputs,
        tvm.target.Target("qcom/adreno-opencl-clml"),
        rpc=rpc,
        load_path="vm_library_clml.so",
    )
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

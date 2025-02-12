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
from tvm import dlight as dl
from tvm.contrib import utils, ndk
from tvm.relax.backend.adreno.clml import OpenCLMLOffLoad


def verify_codegen(clml_mod, clml_codegen):
    source = clml_mod.attrs["external_mods"][0].get_source()
    codegen = json.loads(source)["nodes"]
    for node in range(len(codegen)):
        if codegen[node]["op"] == "input" or codegen[node]["op"] == "const":
            codegen[node]["name"] = ""
        if codegen[node]["op"] == "kernel":
            codegen[node]["name"] = ""
    codegen_str = json.dumps(codegen, sort_keys=True, indent=2)
    known_good_codegen_str = json.dumps(clml_codegen, sort_keys=True, indent=2)
    assert codegen_str == known_good_codegen_str, (
        f"The JSON produced by codegen does not match the expected result. \n"
        f"Actual={codegen_str} \n"
        f"Expected={known_good_codegen_str}"
    )


def build_and_run(
    mod,
    inputs_np,
    target,
    rpc=None,
    params_np={},
    load_path="vm_library.so",
    clml_enable=False,
    clml_codegen=None,
):

    tgt = tvm.target.Target(target, host="llvm -mtriple=aarch64-linux-gnu")
    pipeline = relax.pipeline.get_default_pipeline(tgt)
    mod = pipeline(mod)
    if rpc:
        ex = relax.build(mod, tgt)
        temp = utils.tempdir()
        path = temp.relpath(load_path)
        path = "./" + load_path
        ex.export_library(path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"])
        rpc.upload(path)
        rexec = rpc.load_module(load_path)
        dev = rpc.cl(0)
        vm = relax.VirtualMachine(rexec, dev)
    else:
        ex = relax.build(mod, target)
        dev = tvm.device(target, 0)
        vm = relax.VirtualMachine(ex, dev)

    params_dev = []
    for k, v in params_np.items():
        params_dev.append(tvm.nd.array(v, dev))
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    return tvm_output.numpy()


def run_compare(mod, inputs, params_np, rpc=None, clml_codegen=None):
    clml_mod = copy.deepcopy(mod)
    clml_run_mod = copy.deepcopy(mod)
    mod = tvm.relax.transform.BindParams("main", params_np)(mod)
    clml_mod = tvm.relax.transform.BindParams("main", params_np)(clml_mod)
    clml_run_mod = tvm.relax.transform.BindParams("main", params_np)(clml_run_mod)

    # Verify codegen
    clml_mod = OpenCLMLOffLoad()(clml_mod)
    verify_codegen(clml_mod, clml_codegen)

    # On Mainline CI
    if not rpc:
        return None

    ref = build_and_run(
        mod,
        inputs,
        tvm.target.adreno(),
        rpc=rpc,
        params_np=params_np,
        load_path="vm_library_opencl.so",
    )
    out = build_and_run(
        clml_run_mod,
        inputs,
        tvm.target.adreno(clml=True),
        rpc=rpc,
        params_np=params_np,
        load_path="vm_library_clml.so",
        clml_enable=True,
        clml_codegen=clml_codegen,
    )
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

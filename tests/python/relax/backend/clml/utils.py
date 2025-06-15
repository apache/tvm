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


def build_and_run(
    mod,
    inputs_np,
    target,
    rpc=None,
    load_path="vm_library.so",
    clml_enable=False,
):
    tgt = tvm.target.Target(target, host="llvm -mtriple=aarch64-linux-gnu")
    pipeline = relax.pipeline.get_default_pipeline(tgt)
    mod = pipeline(mod)
    if rpc:
        ex = tvm.compile(mod, tgt)
        temp = utils.tempdir()
        path = temp.relpath(load_path)
        path = "./" + load_path
        ex.export_library(path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"])
        rpc.upload(path)
        rexec = rpc.load_module(load_path)
        dev = rpc.cl(0)
        vm = relax.VirtualMachine(rexec, dev)
    else:
        ex = tvm.compile(mod, target)
        dev = tvm.device(target, 0)
        vm = relax.VirtualMachine(ex, dev)

    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    return tvm_output.numpy()


def run_compare(mod, inputs, params_np, rpc=None):
    clml_mod = copy.deepcopy(mod)
    mod = tvm.relax.transform.BindParams("main", params_np)(mod)
    clml_mod = tvm.relax.transform.BindParams("main", params_np)(clml_mod)

    if not rpc:
        return

    ref = build_and_run(
        mod,
        inputs,
        tvm.target.adreno(),
        rpc=rpc,
        load_path="vm_library_opencl.so",
    )
    out = build_and_run(
        clml_mod,
        inputs,
        tvm.target.adreno(clml=True),
        rpc=rpc,
        load_path="vm_library_clml.so",
        clml_enable=True,
    )
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

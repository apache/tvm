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
"""Test relax vm through rpc."""

import tvm
import numpy as np
from tvm import rpc, relax
from tvm.contrib import utils, tvmjs
from tvm.script import relax as R

proxy_host = "127.0.0.1"
proxy_port = 9090


def get_model():
    pipeline = relax.get_pipeline()

    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(x: R.Tensor([1024], "float32"), y: R.Tensor([1024], "float32")):
            lv0 = R.add(x, y)
            return lv0

    mod = pipeline(Mod)
    sch = tvm.tir.Schedule(mod)
    # manually transform loop
    sch.work_on("add")
    (i,) = sch.get_loops(block=sch.get_block("T_add"))
    i0, i1 = sch.split(i, [None, 128])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")
    return sch.mod


def test_rpc():
    if not tvm.runtime.enabled("rpc"):
        return
    n = 1024
    dtype = "float32"
    temp = utils.tempdir()
    wasm_path = temp.relpath("relax.wasm")
    target = tvm.target.Target("webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm")

    mod = get_model()
    ex = relax.build(mod, target)
    ex.export_library(wasm_path, fcompile=tvmjs.create_tvmjs_wasm)
    wasm_binary = open(wasm_path, "rb").read()

    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )

    def check(remote):
        dev = remote.webgpu(0)
        # invoke the function
        vm = relax.VirtualMachine(remote.system_lib(), device=dev)
        adata = np.random.uniform(size=n).astype(dtype)
        bdata = np.random.uniform(size=n).astype(dtype)
        a = tvm.nd.array(adata, dev)
        b = tvm.nd.array(bdata, dev)
        vm.set_input("main", a, b)
        vm.invoke_stateful("main")
        c = vm.get_outputs("main")
        np.testing.assert_equal(c.numpy(), a.numpy() + b.numpy())

    check(remote)


test_rpc()

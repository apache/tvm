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
"""Simple testcode to test Javascript RPC

To use it, start a rpc proxy with "python -m tvm.exec.rpc_proxy".
Connect javascript end to the websocket port and connect to the RPC.
"""

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils, tvmjs
from tvm.relay.backend import Runtime
import numpy as np

proxy_host = "127.0.0.1"
proxy_port = 9090


def test_rpc():
    if not tvm.runtime.enabled("rpc"):
        return
    # generate the wasm library
    target = tvm.target.Target("webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm")
    runtime = Runtime("cpp", {"system-lib": True})

    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: te.log(te.abs(A(*i) + 1)), name="B")
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]))
    sch = tvm.tir.Schedule(mod)
    (i,) = sch.get_loops(block=sch.get_block("B"))
    i0, i1 = sch.split(i, [None, 32])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")

    fadd = tvm.build(sch.mod, target=target, runtime=runtime)
    temp = utils.tempdir()

    wasm_path = temp.relpath("addone_gpu.wasm")
    fadd.export_library(wasm_path, fcompile=tvmjs.create_tvmjs_wasm)

    wasm_binary = open(wasm_path, "rb").read()
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )

    def check(remote, size):
        # basic function checks.
        dev = remote.webgpu(0)
        adata = np.random.uniform(size=size).astype(A.dtype)
        a = tvm.nd.array(adata, dev)
        b = tvm.nd.array(np.zeros(size, dtype=A.dtype), dev)

        np.testing.assert_equal(a.numpy(), adata)
        f1 = remote.system_lib()
        addone = f1.get_function("main")
        addone(a, b)
        np.testing.assert_allclose(b.numpy(), np.log(np.abs(a.numpy()) + 1), atol=1e-5, rtol=1e-5)
        print("Test pass..")

    check(remote, 71821 * 32)


test_rpc()

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
from tvm.contrib import utils, emcc
import numpy as np

proxy_host = "127.0.0.1"
proxy_port = 9090


def test_rpc():
    if not tvm.runtime.enabled("rpc"):
        return
    # generate the wasm library
    target_device = "webgpu"
    target_host = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target_host):
        raise RuntimeError("Target %s is not enbaled" % target_host)

    n = 2048
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)

    num_thread = 2
    xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
    s[B].bind(xi, te.thread_axis("threadIdx.x"))
    s[B].bind(xo, te.thread_axis("blockIdx.x"))

    fadd = tvm.build(s, [A, B], target_device, target_host=target_host, name="addone")
    temp = utils.tempdir()

    wasm_path = temp.relpath("addone_gpu.wasm")
    fadd.export_library(wasm_path, emcc.create_tvmjs_wasm)

    wasm_binary = open(wasm_path, "rb").read()
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )

    def check(remote):
        # basic function checks.
        dev = remote.webgpu(0)
        adata = np.random.uniform(size=n).astype(A.dtype)
        a = tvm.nd.array(adata, dev)
        b = tvm.nd.array(np.zeros(n, dtype=A.dtype), dev)

        np.testing.assert_equal(a.numpy(), adata)
        f1 = remote.system_lib()
        addone = f1.get_function("addone")
        addone(a, b)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)
        print("Test pass..")

    check(remote)


test_rpc()

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
    runtime = Runtime("cpp", {"system-lib": True})
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)

    fadd = tvm.build(s, [A, B], target, runtime=runtime, name="addone")
    temp = utils.tempdir()

    wasm_path = temp.relpath("addone.wasm")
    fadd.export_library(wasm_path, fcompile=tvmjs.create_tvmjs_wasm)

    wasm_binary = open(wasm_path, "rb").read()

    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )

    def check(remote):
        # basic function checks.
        faddone = remote.get_function("testing.asyncAddOne")
        fecho = remote.get_function("testing.echo")
        assert faddone(100) == 101
        assert fecho(1, 2, 3) == 1
        assert fecho(1, 2, 3) == 1
        assert fecho(100, 2, 3) == 100
        assert fecho("xyz") == "xyz"
        assert bytes(fecho(bytearray(b"123"))) == b"123"
        # run the generated library.
        f1 = remote.system_lib()
        dev = remote.cpu(0)
        a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
        # invoke the function
        addone = f1.get_function("addone")
        addone(a, b)

        # time evaluator
        time_f = f1.time_evaluator("addone", dev, number=100, repeat=10)
        time_f(a, b)
        cost = time_f(a, b).mean
        print("%g secs/op" % cost)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)

    check(remote)


test_rpc()

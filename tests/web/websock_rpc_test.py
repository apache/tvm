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
import os
from tvm import rpc
from tvm.contrib import util, emscripten
import numpy as np

proxy_host = "localhost"
proxy_port = 9090

def test_rpc_array():
    if not tvm.module.enabled("rpc"):
        return
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)
    remote = rpc.connect(proxy_host, proxy_port, key="js")
    target = "llvm -target=asmjs-unknown-emscripten -system-lib"
    def check_remote():
        if not tvm.module.enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        temp = util.tempdir()
        ctx = remote.cpu(0)
        f = tvm.build(s, [A, B], target, name="myadd")
        path_obj = temp.relpath("dev_lib.bc")
        path_dso = temp.relpath("dev_lib.js")
        f.save(path_obj)
        emscripten.create_js(path_dso, path_obj, side_module=True)
        # Upload to suffix as dso so it can be loaded remotely
        remote.upload(path_dso, "dev_lib.dso")
        data = remote.download("dev_lib.dso")
        f1 = remote.load_module("dev_lib.dso")
        a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
        time_f = f1.time_evaluator(f1.entry_name, remote.cpu(0), number=10)
        cost = time_f(a, b).mean
        print('%g secs/op' % cost)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)
    check_remote()

test_rpc_array()

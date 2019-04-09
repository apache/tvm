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
import tvm
from tvm import rpc
from tvm.contrib import util, graph_runtime
import nnvm.symbol as sym
import nnvm.compiler
import numpy as np
import time

def test_rpc_executor():
    host = "localhost"
    port = 9021
    server = rpc.Server(host, port, use_popen=True)
    time.sleep(1)
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.exp(y + x)
    shape = (10, 128)
    dtype = tvm.float32
    shape_dict = {"x": shape, "y": shape}
    tmp = util.tempdir()
    lib_name  = tmp.relpath("net.o")

    graph, lib, _ = nnvm.compiler.build(z, "llvm", shape_dict)
    # save module
    lib.save(lib_name)
    remote = rpc.connect(host, port)
    remote.upload(lib_name)
    ctx = remote.cpu(0)
    # load remote
    rlib = remote.load_module("net.o")

    # Create remotemodule
    m = graph_runtime.create(graph, rlib, remote.cpu(0))
    # get member functions
    set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
    na = tvm.nd.array(np.ones(shape).astype(dtype), ctx)
    nb = tvm.nd.array(np.ones(shape).astype(dtype), ctx)
    # set inputs
    set_input("x", na)
    set_input("y", nb)
    # execute
    run()
    # get outputs
    out = tvm.nd.empty(shape, dtype, ctx)
    get_output(0, out)
    tvm.testing.assert_allclose(
        out.asnumpy(), np.exp(na.asnumpy() + nb.asnumpy()))
    server.terminate()

if __name__ == "__main__":
    test_rpc_executor()

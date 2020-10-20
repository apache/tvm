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
import tvm.testing
from tvm import te
import numpy as np
import json
from tvm import rpc
from tvm.contrib import util, graph_runtime


@tvm.testing.requires_llvm
def test_graph_simple():
    n = 4
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)

    node0 = {"op": "null", "name": "x", "inputs": []}
    node1 = {
        "op": "tvm_op",
        "name": "add",
        "inputs": [[0, 0, 0]],
        "attrs": {"func_name": "myadd", "flatten_data": "1", "num_inputs": "1", "num_outputs": "1"},
    }
    nodes = [node0, node1]
    arg_nodes = [0]
    node_row_ptr = [0, 1, 2]
    outputs = [[1, 0, 0]]
    shape = (4,)
    attrs = {
        "shape": ["list_shape", [shape, shape]],
        "dltype": ["list_str", ["float32", "float32"]],
        "storage_id": ["list_int", [0, 1]],
    }
    graph = {
        "nodes": nodes,
        "arg_nodes": arg_nodes,
        "node_row_ptr": node_row_ptr,
        "heads": outputs,
        "attrs": attrs,
    }
    graph = json.dumps(graph)

    def check_verify():
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        mod = graph_runtime.create(graph, mlib, tvm.cpu(0))
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.run(x=a)
        out = mod.get_output(0, tvm.nd.empty((n,)))
        np.testing.assert_equal(out.asnumpy(), a + 1)

    def check_remote():
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        server = rpc.Server("localhost")
        remote = rpc.connect(server.host, server.port)
        temp = util.tempdir()
        ctx = remote.cpu(0)
        path_dso = temp.relpath("dev_lib.so")
        mlib.export_library(path_dso)
        remote.upload(path_dso)
        mlib = remote.load_module("dev_lib.so")
        mod = graph_runtime.create(graph, mlib, remote.cpu(0))
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.run(x=tvm.nd.array(a, ctx))
        out = tvm.nd.empty((n,), ctx=ctx)
        out = mod.get_output(0, out)
        np.testing.assert_equal(out.asnumpy(), a + 1)

    def check_sharing():
        from tvm import relay

        x = relay.var("x", shape=(1, 10))
        y = relay.var("y", shape=(1, 10))
        z = relay.add(x, y)
        func = relay.Function([x, y], z)

        x_in = np.ones((1, 10)).astype("float32")
        params = {"x": x_in}
        graph, lib, params = relay.build(func, target="llvm", params=params)

        mod_shared = graph_runtime.create(graph, lib, tvm.cpu(0))
        mod_shared.load_params(relay.save_param_dict(params))
        num_mods = 10
        mods = [graph_runtime.create(graph, lib, tvm.cpu(0)) for _ in range(num_mods)]

        for mod in mods:
            mod.share_params(mod_shared, relay.save_param_dict(params))

        a = np.random.uniform(size=(1, 10)).astype("float32")
        for mod in mods:
            mod.run(y=a)
            out = mod.get_output(0, tvm.nd.empty((1, 10)))
            np.testing.assert_equal(out.asnumpy(), x_in + a)

        # Explicitly delete the shared module and verify correctness.
        del mod_shared
        for mod in mods:
            mod.run(y=a)
            out = mod.get_output(0, tvm.nd.empty((1, 10)))
            np.testing.assert_equal(out.asnumpy(), x_in + a)
            del mod

    check_verify()
    check_remote()
    check_sharing()


if __name__ == "__main__":
    test_graph_simple()

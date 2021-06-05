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
import json
import os
import re
import sys
import time

import pytest

import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm import rpc
from tvm.contrib import utils
from tvm.contrib.debugger import debug_executor


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

        def myadd(*args):
            to_return = mlib["myadd"](*args)
            time.sleep(0.25)
            return to_return

        mlib_proxy = tvm.support.FrontendTestModule()
        mlib_proxy["myadd"] = myadd
        try:
            mod = debug_executor.create(graph, mlib_proxy, tvm.cpu(0))
        except ValueError:
            return

        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.set_input(x=a)

        # verify dumproot created
        directory = mod._dump_path
        assert os.path.exists(directory)

        # verify graph is there
        GRAPH_DUMP_FILE_NAME = "_tvmdbg_graph_dump.json"
        assert len(os.listdir(directory)) == 1

        # verify the file name is proper
        graph_dump_path = os.path.join(directory, GRAPH_DUMP_FILE_NAME)
        assert os.path.exists(graph_dump_path)

        # verify the graph contains some expected keys
        with open(graph_dump_path) as graph_f:
            dumped_graph = json.load(graph_f)

        assert isinstance(dumped_graph, dict)
        for k in ("nodes", "arg_nodes", "node_row_ptr", "heads", "attrs"):
            assert k in dumped_graph, f"key {k} not in dumped graph {graph!r}"

        mod.run()
        # Verify the tensors are dumped
        assert len(os.listdir(directory)) > 1

        debug_lines = mod.debug_datum.get_debug_result().split("\n")

        def split_debug_line(i):
            to_return = re.split(r"  [ ]*", debug_lines[i])
            assert to_return[-1] == ""
            to_return = to_return[:-1]  # strip empty trailing part
            return to_return

        assert split_debug_line(0) == [
            "Node Name",
            "Ops",
            "Time(us)",
            "Time(%)",
            "Shape",
            "Inputs",
            "Outputs",
        ]
        myadd_lines = split_debug_line(2)
        assert myadd_lines[0] == "add"
        assert myadd_lines[1] == "myadd"
        runtime_sec = float(myadd_lines[2]) / 1e6  # printed in us

        # Ensure runtime is at least the sleep time and less than a unit prefix order of magnitude.
        # Here we just care that the prefix is correct.
        assert runtime_sec > 0.25 and runtime_sec < 0.25 * 1000

        total_lines = split_debug_line(3)
        assert total_lines[0] == "Total_time"
        assert total_lines[2] == myadd_lines[2]

        CHROME_TRACE_FILE_NAME = "_tvmdbg_execution_trace.json"
        assert os.path.exists(os.path.join(directory, CHROME_TRACE_FILE_NAME))

        with open(os.path.join(directory, CHROME_TRACE_FILE_NAME)) as f:
            trace = json.load(f)
        assert trace["displayTimeUnit"] == "ns"
        events = trace["traceEvents"]
        assert len(events) == 4
        assert all(event["ph"] in ("B", "E") for event in events)
        assert all(event["pid"] == 1 for event in events)
        assert all(event["tid"] == 1 for event in events)
        assert all(event["name"] == "x" for event in events[:2])
        assert all(event["name"] == "add" for event in events[2:])
        assert events[0]["ts"] == 0
        assert events[0]["ph"] == "B"

        # verify the output is correct
        out = mod.get_output(0, tvm.nd.empty((n,)))
        np.testing.assert_equal(out.numpy(), a + 1)

        mod.exit()
        # verify dump root delete after cleanup
        assert not os.path.exists(directory)

    def check_remote():
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        server = rpc.Server("127.0.0.1")
        remote = rpc.connect(server.host, server.port)
        temp = utils.tempdir()
        dev = remote.cpu(0)
        path_dso = temp.relpath("dev_lib.so")
        mlib.export_library(path_dso)
        remote.upload(path_dso)
        mlib = remote.load_module("dev_lib.so")
        try:
            mod = debug_executor.create(graph, mlib, remote.cpu(0))
        except ValueError:
            print("Skip because debug runtime not enabled")
            return
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.run(x=tvm.nd.array(a, dev))
        out = tvm.nd.empty((n,), device=dev)
        out = mod.get_output(0, out)
        np.testing.assert_equal(out.numpy(), a + 1)

    check_verify()
    check_remote()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))

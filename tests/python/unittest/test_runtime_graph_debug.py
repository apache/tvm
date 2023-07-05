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
from distutils.log import debug

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import rpc, te
from tvm._ffi.base import TVMError
from tvm.contrib import utils
from tvm.contrib.debugger import debug_executor
from tvm import relay

# Constants for creating simple graphs, fixtures to avoid free globals
@pytest.fixture
def n():
    return 4


@pytest.fixture
def A(n):
    return te.placeholder((n,), name="A")


@pytest.fixture
def B(A):
    return te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")


@pytest.fixture
def s(B):
    return te.create_schedule(B.op)


@pytest.fixture
def mlib(s, A, B):
    return tvm.build(s, [A, B], "llvm", name="myadd")


@pytest.fixture
def myadd(mlib):
    def _myadd(*args):
        to_return = mlib["myadd"](*args)
        time.sleep(0.25)
        return to_return

    return _myadd


@pytest.fixture
def graph():
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
    return graph


@tvm.testing.requires_llvm
@tvm.testing.requires_rpc
@pytest.mark.skipif(
    tvm.support.libinfo()["USE_PROFILER"] != "ON", reason="TVM was not built with profiler support"
)
def test_end_to_end_graph_simple(graph, n, A, B, s, myadd):
    def check_verify():
        mlib_proxy = tvm.support.FrontendTestModule()
        mlib_proxy["myadd"] = myadd
        mod = debug_executor.create(graph, mlib_proxy, tvm.cpu(0))

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
            "Measurements(us)",
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

    def check_remote(server):
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
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
    check_remote(rpc.Server("127.0.0.1"))


@tvm.testing.requires_llvm
@pytest.mark.skipif(
    tvm.support.libinfo()["USE_PROFILER"] != "ON", reason="TVM was not built with profiler support"
)
def test_run_single_node(graph, n, A, myadd):
    mlib_proxy = tvm.support.FrontendTestModule()
    mlib_proxy["myadd"] = myadd
    mod: debug_executor.GraphModuleDebug = debug_executor.create(graph, mlib_proxy, tvm.cpu(0))

    a = np.random.uniform(size=(n,)).astype(A.dtype)
    mod.set_input(x=a)

    assert len(mod.debug_datum.get_graph_nodes()) == 2
    assert mod.debug_datum.get_graph_nodes()[0]["op"] == "param"
    assert mod.debug_datum.get_graph_nodes()[1]["op"] == "myadd"

    # Running a node with no associated function should return instantly and have 0 runtime
    assert mod.run_individual_node(0, number=1).mean == 0

    # Meanwhile the actual function should take some time, more time if you run it more times
    repeat_1_result = mod.run_individual_node(1, repeat=1)
    assert repeat_1_result.mean > 0

    # Running multiple times (10) should take longer than 1 time
    repeat_3_results = mod.run_individual_node(1, repeat=3)
    assert sum(repeat_3_results.results) > sum(repeat_1_result.results)

    # Increasing the number of repeats should give you the number of results asked for
    assert len(mod.run_individual_node(1, repeat=10).results) == 10

    # Doing repeat_ms should have the run time greater than the asked amount
    start = time.time()
    mod.run_individual_node(1, min_repeat_ms=500)
    end = time.time()
    elapsed_time_in_seconds = end - start
    assert elapsed_time_in_seconds >= 0.5

    # Doing `cooldown_interval_ms` should have the execution time increases
    start = time.time()
    mod.run_individual_node(1, repeat=2, min_repeat_ms=500, cooldown_interval_ms=1000)
    end = time.time()
    elapsed_time_in_seconds_with_def_rep = end - start
    assert elapsed_time_in_seconds_with_def_rep >= 3

    # Doing with `repeats_to_cooldown` not equal 1 should not trigger
    # cooldown after each repeat
    start = time.time()
    mod.run_individual_node(
        1, repeat=2, min_repeat_ms=500, cooldown_interval_ms=1000, repeats_to_cooldown=2
    )
    end = time.time()
    elapsed_time_in_seconds_with_rep_2 = end - start
    assert elapsed_time_in_seconds_with_rep_2 >= 2 and (
        elapsed_time_in_seconds_with_rep_2 < elapsed_time_in_seconds_with_def_rep
    )

    # Going out of bounds of node index throws a tvm error
    with pytest.raises(TVMError):
        mod.run_individual_node(2)


@tvm.testing.requires_llvm
def test_multiple_output():
    x = relay.var("x", shape=(1, 3, 48, 16), dtype="float32")
    t = relay.split(x, [12, 16, 32], 2).astuple()
    x0 = relay.TupleGetItem(t, 0)
    x1 = relay.TupleGetItem(t, 1)
    x2 = relay.TupleGetItem(t, 2)
    x3 = relay.TupleGetItem(t, 3)
    p0 = relay.const(np.random.uniform(-1, 1, (3, 3, 1, 1)).astype("float32"))
    y = relay.nn.conv2d(x2, p0, kernel_size=(1, 1), kernel_layout="OIHW", out_dtype="float32") + x3

    func = relay.Function([x], relay.Tuple([x0, x1, y]))
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    target = tvm.target.Target("llvm")
    device = tvm.cpu()
    lib = relay.build(mod, target=target)
    m = debug_executor.GraphModuleDebug(
        lib["debug_create"]("default", device), [device], lib.get_graph_json(), None
    )
    nodes = m.debug_datum.get_graph_nodes()
    assert nodes[2]["shape"] == [3, 3, 1, 1]


if __name__ == "__main__":
    tvm.testing.main()

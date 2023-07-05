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
import numpy as np
import pytest
from io import StringIO
import csv
import os
import json
import platform

import tvm.testing
import tvm.utils
from tvm.runtime import profiler_vm
from tvm import relay
from tvm.relay.testing import mlp
from tvm.contrib.debugger import debug_executor
from tvm import rpc
from tvm.contrib import utils
from tvm.runtime.profiling import Report
from tvm.script import tir as T


def read_csv(report):
    f = StringIO(report.csv())
    headers = []
    rows = []
    reader = csv.reader(f, delimiter=",")
    # force parsing
    in_header = True
    for row in reader:
        if in_header:
            headers = row
            in_header = False
            rows = [[] for x in headers]
        else:
            for i in range(len(row)):
                rows[i].append(row[i])
    return dict(zip(headers, rows))


@pytest.mark.skipif(not profiler_vm.enabled(), reason="VM Profiler not enabled")
@tvm.testing.skip_if_wheel_test
@tvm.testing.parametrize_targets
def test_vm(target, dev):
    dtype = "float32"
    x = relay.var("x", shape=(relay.Any(), relay.Any()), dtype=dtype)
    y = relay.var("y", shape=(relay.Any(), relay.Any()), dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], relay.add(x, y))
    exe = relay.vm.compile(mod, target)
    vm = profiler_vm.VirtualMachineProfiler(exe, dev)

    data = np.random.rand(28, 28).astype("float32")
    report = vm.profile(data, data, func_name="main")
    assert "fused_add" in str(report)
    assert "Total" in str(report)
    assert "AllocTensorReg" in str(report)
    assert "AllocStorage" in str(report)
    assert report.configuration["Executor"] == "VM"

    csv = read_csv(report)
    assert "Hash" in csv.keys()
    # Ops should have a duration greater than zero.
    assert all(
        [
            float(dur) > 0
            for dur, name in zip(csv["Duration (us)"], csv["Name"])
            if name[:5] == "fused"
        ]
    )
    # AllocTensor or AllocStorage may be cached, so their duration could be 0.
    assert all(
        [
            float(dur) >= 0
            for dur, name in zip(csv["Duration (us)"], csv["Name"])
            if name[:5] != "fused"
        ]
    )


@tvm.testing.parametrize_targets
def test_graph_executor(target, dev):
    mod, params = mlp.get_workload(1)

    exe = relay.build(mod, target, params=params)
    gr = debug_executor.create(exe.get_graph_json(), exe.lib, dev)

    data = np.random.rand(1, 1, 28, 28).astype("float32")
    report = gr.profile(data=data)
    assert "fused_nn_softmax" in str(report)
    assert "Total" in str(report)
    assert "Hash" in str(report)
    assert "Graph" in str(report)


@tvm.testing.parametrize_targets("cuda", "llvm")
@pytest.mark.skipif(
    tvm.get_global_func("runtime.profiling.PAPIMetricCollector", allow_missing=True) is None,
    reason="PAPI profiling not enabled",
)
def test_papi(target, dev):
    target = tvm.target.Target(target)
    if str(target.kind) == "llvm":
        metric = "PAPI_FP_OPS"
    elif str(target.kind) == "cuda":
        metric = "cuda:::event:shared_load:device=0"
    else:
        pytest.skip(f"Target {target.kind} not supported by this test")
    mod, params = mlp.get_workload(1)

    exe = relay.vm.compile(mod, target, params=params)
    vm = profiler_vm.VirtualMachineProfiler(exe, dev)

    data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"), device=dev)
    report = vm.profile(
        data,
        func_name="main",
        collectors=[tvm.runtime.profiling.PAPIMetricCollector({dev: [metric]})],
    )
    assert metric in str(report)

    csv = read_csv(report)
    assert metric in csv.keys()
    assert any([float(x) > 0 for x in csv[metric]])


@tvm.testing.requires_llvm
def test_json():
    mod, params = mlp.get_workload(1)

    exe = relay.vm.compile(mod, "llvm", params=params)
    vm = profiler_vm.VirtualMachineProfiler(exe, tvm.cpu())

    data = np.random.rand(1, 1, 28, 28).astype("float32")
    report = vm.profile(data, func_name="main")
    parsed = json.loads(report.json())
    assert "device_metrics" in parsed
    assert "calls" in parsed
    assert "configuration" in parsed
    assert "Duration (us)" in parsed["calls"][0]
    assert "microseconds" in parsed["calls"][0]["Duration (us)"]
    assert len(parsed["calls"]) > 0
    for call in parsed["calls"]:
        assert isinstance(call["Name"]["string"], str)
        assert isinstance(call["Count"]["count"], int)
        assert isinstance(call["Duration (us)"]["microseconds"], float)


@tvm.testing.requires_llvm
def test_rpc_vm():
    server = rpc.Server(key="profiling")
    remote = rpc.connect("127.0.0.1", server.port, key="profiling")

    mod, params = mlp.get_workload(1)
    exe = relay.vm.compile(mod, "llvm", params=params)
    temp = utils.tempdir()
    path = temp.relpath("lib.tar")
    exe.mod.export_library(path)
    remote.upload(path)
    rexec = remote.load_module("lib.tar")
    vm = profiler_vm.VirtualMachineProfiler(rexec, remote.cpu())
    report = vm.profile(tvm.nd.array(np.ones((1, 1, 28, 28), dtype="float32"), device=remote.cpu()))
    assert len(report.calls) > 0


def test_rpc_graph():
    server = rpc.Server(key="profiling")
    remote = rpc.connect("127.0.0.1", server.port, key="profiling")

    mod, params = mlp.get_workload(1)
    exe = relay.build(mod, "llvm", params=params)
    temp = utils.tempdir()
    path = temp.relpath("lib.tar")
    exe.export_library(path)
    remote.upload(path)
    rexec = remote.load_module("lib.tar")

    gr = debug_executor.create(exe.get_graph_json(), rexec, remote.cpu())

    data = np.random.rand(1, 1, 28, 28).astype("float32")
    report = gr.profile(data=data)
    assert len(report.calls) > 0


def test_report_serialization():
    mod, params = mlp.get_workload(1)

    exe = relay.vm.compile(mod, "llvm", params=params)
    vm = profiler_vm.VirtualMachineProfiler(exe, tvm.cpu())

    data = np.random.rand(1, 1, 28, 28).astype("float32")
    report = vm.profile(data, func_name="main")

    report2 = Report.from_json(report.json())
    # Equality on reports compares pointers, so we compare the printed
    # results instead.

    # Use .table() instead of str(), because str() includes aggregate
    # and column summations whose values may be impacted by otherwise
    # negligible conversion errors. (2 occurrences / 3000 trials)
    assert report.table(aggregate=False, col_sums=False) == report2.table(
        aggregate=False, col_sums=False
    )


@T.prim_func
def axpy_cpu(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [10], "float64")
    B = T.match_buffer(b, [10], "float64")
    C = T.match_buffer(c, [10], "float64")
    for i in range(10):
        C[i] = A[i] + B[i]


@T.prim_func
def axpy_gpu(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [10], "float64")
    B = T.match_buffer(b, [10], "float64")
    C = T.match_buffer(c, [10], "float64")
    for i in T.thread_binding(0, 10, "threadIdx.x"):
        C[i] = A[i] + B[i]


@tvm.testing.parametrize_targets("cuda", "llvm")
@pytest.mark.skipif(
    tvm.get_global_func("runtime.profiling.PAPIMetricCollector", allow_missing=True) is None,
    reason="PAPI profiling not enabled",
)
def test_profile_function(target, dev):
    target = tvm.target.Target(target)
    if str(target.kind) == "llvm":
        metric = "PAPI_FP_OPS"
        func = axpy_cpu
    elif str(target.kind) == "cuda":
        metric = (
            "cuda:::gpu__compute_memory_access_throughput.max.pct_of_peak_sustained_region:device=0"
        )
        func = axpy_gpu
    else:
        pytest.skip(f"Target {target.kind} not supported by this test")
    f = tvm.build(func, target=target)
    a = tvm.nd.array(np.ones(10), device=dev)
    b = tvm.nd.array(np.ones(10), device=dev)
    c = tvm.nd.array(np.zeros(10), device=dev)
    report = tvm.runtime.profiling.profile_function(
        f, dev, [tvm.runtime.profiling.PAPIMetricCollector({dev: [metric]})]
    )(a, b, c)
    assert metric in report.keys()
    assert report[metric].value > 0


if __name__ == "__main__":
    tvm.testing.main()

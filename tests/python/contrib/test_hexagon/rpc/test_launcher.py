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

import sys
import pytest
import numpy as np
import os

import tvm.testing
from tvm import te
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib import utils, ndk
from tvm.contrib.hexagon.build import HexagonLauncher
import tvm.contrib.hexagon.hexagon as hexagon

from conftest import requires_rpc_tracker, requires_ndk_cc


@requires_rpc_tracker
@requires_ndk_cc
def test_rpc_on_android(tvm_tracker_host, tvm_tracker_port, android_serial_number):
    if "TVM_NDK_CC" not in os.environ:
        raise RuntimeError(
            "Require environment variable TVM_NDK_CC" " to be the NDK standalone compiler"
        )
    target = "llvm -mtriple=arm64-linux-android"

    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    a_np = np.random.uniform(size=1024).astype(A.dtype)
    temp = utils.tempdir()

    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=64)
    s[B].parallel(xi)
    s[B].pragma(xo, "parallel_launch_point")
    s[B].pragma(xi, "parallel_barrier_when_finish")
    f = tvm.build(s, [A, B], target, name="myadd_cpu")
    path_dso_cpu = temp.relpath("cpu_lib.so")
    f.export_library(path_dso_cpu, ndk.create_shared)

    launcher = HexagonLauncher(serial_number=android_serial_number)
    launcher.android_run_rpc(rpc_tracker_host=tvm_tracker_host, rpc_tracker_port=tvm_tracker_port)

    remote_kw = {
        "host": tvm_tracker_host,
        "port": tvm_tracker_port,
        "priority": 0,
        "timeout": 60,
    }
    launcher.android_remote_setup(remote_kw)

    print("Run CPU test ...")
    remote = launcher.android_remote
    dev = remote.cpu(0)
    launcher.upload(path_dso_cpu)
    f2 = remote.load_module("cpu_lib.so")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
    time_f = f2.time_evaluator(f2.entry_name, dev, number=10)
    cost = time_f(a, b).mean
    print("%g secs/op\n" % cost)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)
    launcher.close()


@requires_rpc_tracker
def test_add(tvm_tracker_host, tvm_tracker_port, android_serial_number):
    dtype = "int8"
    A = tvm.te.placeholder((2,), dtype=dtype)
    B = tvm.te.placeholder((1,), dtype=dtype)
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)

    target = tvm.target.hexagon("v68", link_params=True)
    func = tvm.build(sched, [A, B, C], target=target, target_host=target, name="add")

    temp = utils.tempdir()
    dso_binary = "test_binary.so"
    dso_binary_path = temp.relpath(dso_binary)
    func.save(dso_binary_path)

    launcher = HexagonLauncher(serial_number=android_serial_number)
    launcher.android_run_rpc(rpc_tracker_host=tvm_tracker_host, rpc_tracker_port=tvm_tracker_port)
    launcher.hexagon_setup()
    remote_kw = {
        "host": tvm_tracker_host,
        "port": tvm_tracker_port,
        "priority": 0,
        "timeout": 60,
    }
    launcher.android_remote_setup(remote_kw)
    launcher.hexagon_session_setup(remote_kw)
    launcher.upload(dso_binary_path, dso_binary)

    with launcher.session as sess:
        mod = launcher.get_module(dso_binary)
        A_data = tvm.nd.array(np.array([2, 3], dtype=dtype), device=sess.device)
        assert (A_data.numpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype=dtype), device=sess.device)
        assert (B_data.numpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype=dtype), device=sess.device)
        assert (C_data.numpy() == np.array([0, 0])).all()

        mod["add"](A_data, B_data, C_data)
        assert (C_data.numpy() == np.array([6, 7])).all()
    launcher.close()


class TestMatMul:
    M = tvm.testing.parameter(32)
    N = tvm.testing.parameter(32)
    K = tvm.testing.parameter(32)

    @requires_rpc_tracker
    def test_matmul(self, tvm_tracker_host, tvm_tracker_port, android_serial_number, M, N, K):
        X = te.placeholder((M, K), dtype="float32")
        Y = te.placeholder((K, N), dtype="float32")
        k1 = te.reduce_axis((0, K), name="k1")
        Z = te.compute((M, N), lambda i, j: te.sum(X[i, k1] * Y[k1, j], axis=[k1]))
        schedule = te.create_schedule(Z.op)

        target_hexagon = tvm.target.hexagon("v68", link_params=True)
        func = tvm.build(schedule, [X, Y, Z], target=target_hexagon, target_host=target_hexagon)

        temp = utils.tempdir()
        dso_binary = "test_binary.so"
        dso_binary_path = temp.relpath(dso_binary)
        func.save(dso_binary_path)

        launcher = HexagonLauncher(serial_number=android_serial_number)
        launcher.android_run_rpc(
            rpc_tracker_host=tvm_tracker_host, rpc_tracker_port=tvm_tracker_port
        )
        launcher.hexagon_setup()
        remote_kw = {
            "host": tvm_tracker_host,
            "port": tvm_tracker_port,
            "priority": 0,
            "timeout": 60,
        }
        launcher.android_remote_setup(remote_kw)
        launcher.hexagon_session_setup(remote_kw)
        launcher.upload(dso_binary_path, dso_binary)

        x = np.random.uniform(size=[i.value for i in X.shape]).astype(X.dtype)
        y = np.random.uniform(size=[i.value for i in Y.shape]).astype(Y.dtype)
        z = np.zeros([i.value for i in Z.shape], dtype=Z.dtype)

        with launcher.session as sess:
            mod = launcher.get_module(dso_binary)
            xt = tvm.nd.array(x, device=sess.device)
            yt = tvm.nd.array(y, device=sess.device)
            zt = tvm.nd.array(z, device=sess.device)
            mod(xt, yt, zt)

        target_llvm = tvm.target.Target("llvm")
        mod = tvm.build(schedule, [X, Y, Z], target=target_llvm, target_host=target_llvm)
        device = tvm.cpu(0)
        xtcpu = tvm.nd.array(x, device)
        ytcpu = tvm.nd.array(y, device)
        ztcpu = tvm.nd.array(z, device)
        mod(xtcpu, ytcpu, ztcpu)
        launcher.close()

        tvm.testing.assert_allclose(zt.asnumpy(), ztcpu.asnumpy(), rtol=1e-4)


@requires_rpc_tracker
def test_graph_executor(tvm_tracker_host, tvm_tracker_port, android_serial_number):
    dtype = "float32"
    data = relay.var("data", relay.TensorType((1, 64, 64, 3), dtype))
    weight = relay.var("weight", relay.TensorType((5, 5, 3, 8), dtype))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = relay.transform.InferType()(relay_mod)

    target_hexagon = tvm.target.hexagon("v68")
    runtime = Runtime("cpp")
    executor = Executor("graph")

    temp = utils.tempdir()
    dso_binary = "test_binary.so"
    dso_binary_path = temp.relpath(dso_binary)

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            target=target_hexagon,
            target_host=target_hexagon,
            runtime=runtime,
            executor=executor,
        )
        lowered.get_lib().save(dso_binary_path)

    launcher = HexagonLauncher(serial_number=android_serial_number)
    launcher.android_run_rpc(rpc_tracker_host=tvm_tracker_host, rpc_tracker_port=tvm_tracker_port)
    launcher.hexagon_setup()
    remote_kw = {
        "host": tvm_tracker_host,
        "port": tvm_tracker_port,
        "priority": 0,
        "timeout": 60,
    }
    launcher.android_remote_setup(remote_kw)
    launcher.hexagon_session_setup(remote_kw)
    launcher.upload(dso_binary_path, dso_binary)

    graph_mod = launcher.get_local_graph_executor(lowered, dso_binary)
    weight_in = np.random.rand(5, 5, 3, 8).astype(dtype=dtype)
    data_in = np.random.rand(1, 64, 64, 3).astype(dtype=dtype)
    graph_mod.set_input(weight=weight_in)
    graph_mod.run(data=data_in)
    hexagon_output = graph_mod.get_output(0).numpy()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            target=target_llvm,
            target_host=target_llvm,
            runtime=runtime,
            executor=executor,
        )
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(weight=weight_in)
    llvm_graph_mod.run(data=data_in)
    expected_output = llvm_graph_mod.get_output(0).numpy()
    launcher.close()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))

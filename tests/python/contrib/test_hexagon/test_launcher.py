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

import pathlib
import sys
import pytest
import numpy as np
import logging

import tvm.testing
from tvm import te
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib import utils, ndk
from tvm.contrib.hexagon.build import HexagonLauncher
import tvm.contrib.hexagon.hexagon as hexagon

from .conftest import requires_hexagon_toolchain


@requires_hexagon_toolchain
def test_add(android_serial_number, tvm_tracker_host, tvm_tracker_port):
    dtype = "int8"
    A = tvm.te.placeholder((2,), dtype=dtype)
    B = tvm.te.placeholder((1,), dtype=dtype)
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)

    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    func = tvm.build(
        sched, [A, B, C], tvm.target.Target(target_hexagon, host=target_hexagon), name="add"
    )

    temp = utils.tempdir()
    dso_binary = "test_binary.so"
    dso_binary_path = temp.relpath(dso_binary)
    func.save(dso_binary_path)

    if not android_serial_number:
        pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    rpc_info = {
        "rpc_tracker_host": tvm_tracker_host,
        "rpc_tracker_port": tvm_tracker_port,
        "rpc_server_port": 7070,
    }
    launcher = HexagonLauncher(serial_number=android_serial_number, rpc_info=rpc_info)
    launcher.upload(dso_binary_path, dso_binary)
    launcher.start_server()

    with launcher.start_session() as sess:
        mod = launcher.load_module(dso_binary, sess)
        A_data = tvm.nd.array(np.array([2, 3], dtype=dtype), device=sess.device)
        assert (A_data.numpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype=dtype), device=sess.device)
        assert (B_data.numpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype=dtype), device=sess.device)
        assert (C_data.numpy() == np.array([0, 0])).all()

        mod["add"](A_data, B_data, C_data)
        assert (C_data.numpy() == np.array([6, 7])).all()
    launcher.stop_server()


@requires_hexagon_toolchain
def test_add_vtcm(android_serial_number, tvm_tracker_host, tvm_tracker_port):
    dtype = "int8"
    A = tvm.te.placeholder((2,), dtype=dtype)
    B = tvm.te.placeholder((1,), dtype=dtype)
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)

    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    func = tvm.build(
        sched, [A, B, C], tvm.target.Target(target_hexagon, host=target_hexagon), name="add"
    )

    temp = utils.tempdir()
    dso_binary = "test_binary.so"
    dso_binary_path = temp.relpath(dso_binary)
    func.save(dso_binary_path)

    if not android_serial_number:
        pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    rpc_info = {
        "rpc_tracker_host": tvm_tracker_host,
        "rpc_tracker_port": tvm_tracker_port,
        "rpc_server_port": 7070,
    }
    launcher = HexagonLauncher(serial_number=android_serial_number, rpc_info=rpc_info)
    launcher.upload(dso_binary_path, dso_binary)
    launcher.start_server()

    with launcher.start_session() as sess:
        mod = launcher.load_module(dso_binary, sess)
        A_data = tvm.nd.empty(A.shape, A.dtype, sess.device, "global.vtcm")
        A_data.copyfrom(np.array([2, 3]))

        B_data = tvm.nd.empty(B.shape, B.dtype, sess.device, "global.vtcm")
        B_data.copyfrom(np.array([4]))

        C_data = tvm.nd.empty(C.shape, C.dtype, sess.device, "global.vtcm")
        C_data.copyfrom(np.array([0, 0]))

        mod["add"](A_data, B_data, C_data)
        result = C_data.numpy()
        assert (result == np.array([6, 7])).all()
    launcher.stop_server()


class TestMatMul:
    M = tvm.testing.parameter(32)
    N = tvm.testing.parameter(32)
    K = tvm.testing.parameter(32)

    @requires_hexagon_toolchain
    def test_matmul(self, android_serial_number, tvm_tracker_host, tvm_tracker_port, M, N, K):
        X = te.placeholder((M, K), dtype="float32")
        Y = te.placeholder((K, N), dtype="float32")
        k1 = te.reduce_axis((0, K), name="k1")
        Z = te.compute((M, N), lambda i, j: te.sum(X[i, k1] * Y[k1, j], axis=[k1]))
        schedule = te.create_schedule(Z.op)

        target_hexagon = tvm.target.hexagon("v68", link_params=True)
        func = tvm.build(
            schedule, [X, Y, Z], tvm.target.Target(target_hexagon, host=target_hexagon)
        )

        temp = utils.tempdir()
        dso_binary = "test_binary.so"
        dso_binary_path = temp.relpath(dso_binary)
        func.save(dso_binary_path)

        if not android_serial_number:
            pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

        rpc_info = {
            "rpc_tracker_host": tvm_tracker_host,
            "rpc_tracker_port": tvm_tracker_port,
            "rpc_server_port": 7070,
        }
        launcher = HexagonLauncher(serial_number=android_serial_number, rpc_info=rpc_info)
        launcher.upload(dso_binary_path, dso_binary)
        launcher.start_server()

        x = np.random.uniform(size=[i.value for i in X.shape]).astype(X.dtype)
        y = np.random.uniform(size=[i.value for i in Y.shape]).astype(Y.dtype)
        z = np.zeros([i.value for i in Z.shape], dtype=Z.dtype)

        with launcher.start_session() as sess:
            mod = launcher.load_module(dso_binary, sess)
            xt = tvm.nd.array(x, device=sess.device)
            yt = tvm.nd.array(y, device=sess.device)
            zt = tvm.nd.array(z, device=sess.device)
            mod(xt, yt, zt)

        launcher.stop_server()

        target_llvm = tvm.target.Target("llvm")
        mod = tvm.build(schedule, [X, Y, Z], tvm.target.Target(target_llvm, host=target_llvm))
        device = tvm.cpu(0)
        xtcpu = tvm.nd.array(x, device)
        ytcpu = tvm.nd.array(y, device)
        ztcpu = tvm.nd.array(z, device)
        mod(xtcpu, ytcpu, ztcpu)

        tvm.testing.assert_allclose(zt.numpy(), ztcpu.numpy(), rtol=1e-4)


@requires_hexagon_toolchain
def test_graph_executor(android_serial_number, tvm_tracker_host, tvm_tracker_port):
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

    weight_in = np.random.rand(5, 5, 3, 8).astype(dtype=dtype)
    data_in = np.random.rand(1, 64, 64, 3).astype(dtype=dtype)
    params = {"weight": weight_in}
    inputs = {"data": data_in}

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            runtime=runtime,
            executor=executor,
        )
        lowered.get_lib().save(dso_binary_path)

    if not android_serial_number:
        pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    rpc_info = {
        "rpc_tracker_host": tvm_tracker_host,
        "rpc_tracker_port": tvm_tracker_port,
        "rpc_server_port": 7070,
    }
    launcher = HexagonLauncher(serial_number=android_serial_number, rpc_info=rpc_info)
    launcher.upload(dso_binary_path, dso_binary)
    launcher.start_server()

    with launcher.start_session() as sess:
        graph_mod = launcher.get_graph_executor(lowered.get_graph_json(), dso_binary, sess)
        graph_mod.set_input(**params)
        graph_mod.run(**inputs)
        hexagon_output = graph_mod.get_output(0).numpy()
        launcher.stop_server()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=runtime,
            executor=executor,
        )
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**params)
    llvm_graph_mod.run(**inputs)
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


@requires_hexagon_toolchain
def test_graph_executor_multiple_conv2d(tvm_tracker_host, tvm_tracker_port, android_serial_number):
    dtype = "float32"
    input_shape = (1, 8, 8, 3)
    w1_shape = (5, 5, 3, 1)
    w2_shape = (5, 5, 1, 3)
    data = relay.var("data", relay.TensorType(input_shape, dtype))
    weight1 = relay.var("weight1", relay.TensorType(w1_shape, dtype))
    weight2 = relay.var("weight2", relay.TensorType(w2_shape, dtype))
    y1 = relay.nn.conv2d(
        data,
        weight1,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    y2 = relay.nn.conv2d(
        y1,
        weight2,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight1, weight2], y2)
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
            tvm.target.Target(target_hexagon, host=target_hexagon),
            runtime=runtime,
            executor=executor,
        )
        lowered.get_lib().save(dso_binary_path)

    if not android_serial_number:
        pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    rpc_info = {
        "rpc_tracker_host": tvm_tracker_host,
        "rpc_tracker_port": tvm_tracker_port,
        "rpc_server_port": 7070,
    }
    launcher = HexagonLauncher(serial_number=android_serial_number, rpc_info=rpc_info)
    launcher.upload(dso_binary_path, dso_binary)
    launcher.start_server()

    weight1_data = np.random.rand(w1_shape[0], w1_shape[1], w1_shape[2], w1_shape[3]).astype(
        dtype=dtype
    )
    weight2_data = np.random.rand(w2_shape[0], w2_shape[1], w2_shape[2], w2_shape[3]).astype(
        dtype=dtype
    )
    input_data = np.random.rand(
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    ).astype(dtype=dtype)

    params = {"weight1": weight1_data, "weight2": weight2_data}
    inputs = {"data": input_data}

    with launcher.start_session() as sess:
        graph_mod = launcher.get_graph_executor(lowered.get_graph_json(), dso_binary, sess)
        graph_mod.set_input(**params)
        graph_mod.run(**inputs)
        hexagon_output = graph_mod.get_output(0).numpy()
        launcher.stop_server()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=runtime,
            executor=executor,
        )
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**params)
    llvm_graph_mod.run(**inputs)
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


@requires_hexagon_toolchain
def test_aot_executor(tvm_tracker_host, tvm_tracker_port, android_serial_number):
    dtype = "float32"
    input_shape = (1, 128, 128, 3)
    w_shape = (5, 5, 3, 8)
    data = relay.var("data", relay.TensorType(input_shape, dtype))
    weight = relay.var("weight", relay.TensorType(w_shape, dtype))
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
    temp = utils.tempdir()
    dso_binary = "test_binary.so"
    dso_binary_path = temp / dso_binary

    weight_data = np.random.rand(w_shape[0], w_shape[1], w_shape[2], w_shape[3]).astype(dtype=dtype)
    input_data = np.random.rand(
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    ).astype(dtype=dtype)

    params = {"weight": weight_data}
    inputs = {"data": input_data}

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            params=params,
            target=tvm.target.Target(target_hexagon, host="c"),
            runtime=Runtime("cpp"),
            executor=Executor("aot", {"unpacked-api": False, "interface-api": "c"}),
        )
        lowered.export_library(
            dso_binary_path, fcompile=hexagon.create_aot_shared, hexagon_arch="v68"
        )

    if not android_serial_number:
        pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    rpc_info = {
        "rpc_tracker_host": tvm_tracker_host,
        "rpc_tracker_port": tvm_tracker_port,
        "rpc_server_port": 7070,
    }
    launcher = HexagonLauncher(serial_number=android_serial_number, rpc_info=rpc_info)
    launcher.upload(dso_binary_path, dso_binary)
    launcher.start_server()

    with launcher.start_session() as sess:
        aot_mod = launcher.get_aot_executor(dso_binary, sess)
        aot_mod.set_input(**inputs)
        aot_mod.run()
        hexagon_output = aot_mod.get_output(0).numpy()
        launcher.stop_server()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=Runtime("cpp"),
            executor=Executor("graph"),
        )

    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**params)
    llvm_graph_mod.run(**inputs)
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


@requires_hexagon_toolchain
def test_aot_executor_multiple_conv2d(tvm_tracker_host, tvm_tracker_port, android_serial_number):
    dtype = "float32"
    input_shape = (1, 8, 8, 3)
    w1_shape = (5, 5, 3, 1)
    w2_shape = (5, 5, 1, 3)
    data = relay.var("data", relay.TensorType(input_shape, dtype))
    weight1 = relay.var("weight1", relay.TensorType(w1_shape, dtype))
    weight2 = relay.var("weight2", relay.TensorType(w2_shape, dtype))
    y1 = relay.nn.conv2d(
        data,
        weight1,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    y2 = relay.nn.conv2d(
        y1,
        weight2,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight1, weight2], y2)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = relay.transform.InferType()(relay_mod)

    target_hexagon = tvm.target.hexagon("v68")
    temp = utils.tempdir()
    dso_binary = "test_binary.so"
    dso_binary_path = temp / dso_binary

    weight1_data = np.random.rand(w1_shape[0], w1_shape[1], w1_shape[2], w1_shape[3]).astype(
        dtype=dtype
    )
    weight2_data = np.random.rand(w2_shape[0], w2_shape[1], w2_shape[2], w2_shape[3]).astype(
        dtype=dtype
    )
    input_data = np.random.rand(
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    ).astype(dtype=dtype)

    params = {"weight1": weight1_data, "weight2": weight2_data}
    inputs = {"data": input_data}

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            params=params,
            target=tvm.target.Target(target_hexagon, host="c"),
            runtime=Runtime("cpp"),
            executor=Executor("aot", {"unpacked-api": False, "interface-api": "c"}),
        )
        lowered.export_library(
            dso_binary_path, fcompile=hexagon.create_aot_shared, hexagon_arch="v68"
        )

    if not android_serial_number:
        pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    rpc_info = {
        "rpc_tracker_host": tvm_tracker_host,
        "rpc_tracker_port": tvm_tracker_port,
        "rpc_server_port": 7070,
    }
    launcher = HexagonLauncher(serial_number=android_serial_number, rpc_info=rpc_info)
    launcher.upload(dso_binary_path, dso_binary)
    launcher.start_server()

    with launcher.start_session() as sess:
        aot_mod = launcher.get_aot_executor(dso_binary, sess)
        aot_mod.set_input(**inputs)
        aot_mod.run()
        hexagon_output = aot_mod.get_output(0).numpy()
        launcher.stop_server()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=Runtime("cpp"),
            executor=Executor("graph"),
        )

    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**params)
    llvm_graph_mod.run(**inputs)
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))

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
# pylint: disable=invalid-name,missing-function-docstring,redefined-outer-name
""" Test rpc based launcher for hexagon """
import pytest

import numpy as np

import tvm.testing
from tvm import relay, te
from tvm.contrib.hexagon.session import Session
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.hexagon.build import HexagonLauncherRPC
from tvm.contrib.hexagon.hexagon_profiler import HexagonProfiler

from .infrastructure import get_hexagon_target


@tvm.testing.requires_hexagon
def test_add(hexagon_session: Session):
    """Test simple add"""
    dtype = "int8"
    placeholder_a = tvm.te.placeholder((2,), dtype=dtype)
    placeholder_b = tvm.te.placeholder((1,), dtype=dtype)
    compute_c = tvm.te.compute(
        placeholder_a.shape, lambda i: placeholder_a[i] + placeholder_b[0], name="C"
    )
    sched = tvm.te.create_schedule(compute_c.op)

    func = tvm.build(
        sched,
        [placeholder_a, placeholder_b, compute_c],
        get_hexagon_target("v68"),
        name="add",
    )

    mod = hexagon_session.load_module(func)

    a_data = tvm.nd.array(np.array([2, 3], dtype=dtype), device=hexagon_session.device)
    assert (a_data.numpy() == np.array([2, 3])).all()
    b_data = tvm.nd.array(np.array([4], dtype=dtype), device=hexagon_session.device)
    assert (b_data.numpy() == np.array([4])).all()
    c_data = tvm.nd.array(np.array([0, 0], dtype=dtype), device=hexagon_session.device)
    assert (c_data.numpy() == np.array([0, 0])).all()
    mod["add"](a_data, b_data, c_data)
    assert (c_data.numpy() == np.array([6, 7])).all()


@tvm.testing.requires_hexagon
def test_add_vtcm(hexagon_session: Session):
    """Test add on VTCM"""
    dtype = "int8"
    placeholder_a = tvm.te.placeholder((2,), dtype=dtype)
    placeholder_b = tvm.te.placeholder((1,), dtype=dtype)
    compute_c = tvm.te.compute(
        placeholder_a.shape, lambda i: placeholder_a[i] + placeholder_b[0], name="C"
    )
    sched = tvm.te.create_schedule(compute_c.op)

    func = tvm.build(
        sched,
        [placeholder_a, placeholder_b, compute_c],
        get_hexagon_target("v68"),
        name="add",
    )

    mod = hexagon_session.load_module(func)

    a_data = tvm.nd.empty(
        placeholder_a.shape, placeholder_a.dtype, hexagon_session.device, "global.vtcm"
    )
    a_data.copyfrom(np.array([2, 3]))

    b_data = tvm.nd.empty(
        placeholder_b.shape, placeholder_b.dtype, hexagon_session.device, "global.vtcm"
    )
    b_data.copyfrom(np.array([4]))

    c_data = tvm.nd.empty(compute_c.shape, compute_c.dtype, hexagon_session.device, "global.vtcm")
    c_data.copyfrom(np.array([0, 0]))

    mod["add"](a_data, b_data, c_data)
    result = c_data.numpy()
    assert (result == np.array([6, 7])).all()


class TestMatMul:
    """Test matmul class"""

    size_m = tvm.testing.parameter(32)
    size_n = tvm.testing.parameter(32)
    size_k = tvm.testing.parameter(32)

    @tvm.testing.requires_hexagon
    def test_matmul(self, hexagon_session, size_m, size_n, size_k):
        """Test matmul"""
        placeholder_x = te.placeholder((size_m, size_k), dtype="float32")
        placeholder_y = te.placeholder((size_k, size_n), dtype="float32")
        reduce_k1 = te.reduce_axis((0, size_k), name="k1")
        compute_z = te.compute(
            (size_m, size_n),
            lambda i, j: te.sum(
                placeholder_x[i, reduce_k1] * placeholder_y[reduce_k1, j], axis=[reduce_k1]
            ),
        )
        schedule = te.create_schedule(compute_z.op)

        func = tvm.build(
            schedule,
            [placeholder_x, placeholder_y, compute_z],
            get_hexagon_target("v68"),
        )

        mod = hexagon_session.load_module(func)

        x_data = np.random.uniform(size=[i.value for i in placeholder_x.shape]).astype(
            placeholder_x.dtype
        )
        y_data = np.random.uniform(size=[i.value for i in placeholder_y.shape]).astype(
            placeholder_y.dtype
        )
        z_data = np.zeros([i.value for i in compute_z.shape], dtype=compute_z.dtype)

        x_array = tvm.nd.array(x_data, device=hexagon_session.device)
        y_array = tvm.nd.array(y_data, device=hexagon_session.device)
        z_array = tvm.nd.array(z_data, device=hexagon_session.device)
        mod(x_array, y_array, z_array)

        target_llvm = tvm.target.Target("llvm")
        mod = tvm.build(
            schedule,
            [placeholder_x, placeholder_y, compute_z],
            tvm.target.Target(target_llvm, host=target_llvm),
        )
        device = tvm.cpu(0)
        xtcpu = tvm.nd.array(x_data, device)
        ytcpu = tvm.nd.array(y_data, device)
        ztcpu = tvm.nd.array(z_data, device)
        mod(xtcpu, ytcpu, ztcpu)

        tvm.testing.assert_allclose(z_array.numpy(), ztcpu.numpy(), rtol=1e-4)


@tvm.testing.requires_hexagon
def test_graph_executor(hexagon_session: Session):
    """Test graph executor"""
    dtype = "float32"
    data = relay.var("data", relay.TensorType((1, 64, 64, 3), dtype))
    weight = relay.var("weight", relay.TensorType((5, 5, 3, 8), dtype))
    conv2d_op = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], conv2d_op)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = relay.transform.InferType()(relay_mod)

    runtime = Runtime("cpp")
    executor = Executor("graph")

    weight_in = np.random.rand(5, 5, 3, 8).astype(dtype=dtype)
    data_in = np.random.rand(1, 64, 64, 3).astype(dtype=dtype)
    params = {"weight": weight_in}
    inputs = {"data": data_in}

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            get_hexagon_target("v68"),
            runtime=runtime,
            executor=executor,
        )

    graph_mod = hexagon_session.get_executor_from_factory(lowered)
    graph_mod.set_input(**params)
    graph_mod.run(**inputs)
    hexagon_output = graph_mod.get_output(0).numpy()

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


@tvm.testing.requires_hexagon
def test_graph_executor_multiple_conv2d(hexagon_session: Session):
    """Test multiple conv2d nodes with graph_executor"""
    dtype = "float32"
    input_shape = (1, 8, 8, 3)
    w1_shape = (5, 5, 3, 1)
    w2_shape = (5, 5, 1, 3)
    data = relay.var("data", relay.TensorType(input_shape, dtype))
    weight1 = relay.var("weight1", relay.TensorType(w1_shape, dtype))
    weight2 = relay.var("weight2", relay.TensorType(w2_shape, dtype))
    conv2d_op1 = relay.nn.conv2d(
        data,
        weight1,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    conv2d_op2 = relay.nn.conv2d(
        conv2d_op1,
        weight2,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight1, weight2], conv2d_op2)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = relay.transform.InferType()(relay_mod)

    runtime = Runtime("cpp")
    executor = Executor("graph")

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            get_hexagon_target("v68"),
            runtime=runtime,
            executor=executor,
        )

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

    graph_mod = hexagon_session.get_executor_from_factory(lowered)
    graph_mod.set_input(**params)
    graph_mod.run(**inputs)
    hexagon_output = graph_mod.get_output(0).numpy()

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


@tvm.testing.requires_hexagon
def test_aot_executor(hexagon_session: Session, aot_host_target, aot_target):
    """Test AOT executor"""
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
            target=tvm.target.Target(aot_target, host=aot_host_target),
            runtime=Runtime("cpp"),
            executor=Executor("aot", {"unpacked-api": False, "interface-api": "packed"}),
        )

    aot_mod = hexagon_session.get_executor_from_factory(lowered)
    aot_mod.set_input(**inputs)
    aot_mod.run()
    hexagon_output = aot_mod.get_output(0).numpy()

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


@tvm.testing.requires_hexagon
def test_aot_executor_multiple_conv2d(hexagon_session: Session, aot_host_target, aot_target):
    """Test multiple conv2d nodes with AOT executor"""
    dtype = "float32"
    input_shape = (1, 8, 8, 3)
    w1_shape = (5, 5, 3, 1)
    w2_shape = (5, 5, 1, 3)
    data = relay.var("data", relay.TensorType(input_shape, dtype))
    weight1 = relay.var("weight1", relay.TensorType(w1_shape, dtype))
    weight2 = relay.var("weight2", relay.TensorType(w2_shape, dtype))
    conv2d_op1 = relay.nn.conv2d(
        data,
        weight1,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    conv2d_op2 = relay.nn.conv2d(
        conv2d_op1,
        weight2,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight1, weight2], conv2d_op2)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = relay.transform.InferType()(relay_mod)

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
            target=tvm.target.Target(aot_target, host=aot_host_target),
            runtime=Runtime("cpp"),
            executor=Executor("aot", {"unpacked-api": False, "interface-api": "packed"}),
        )

    aot_mod = hexagon_session.get_executor_from_factory(lowered)
    aot_mod.set_input(**inputs)
    aot_mod.run()
    hexagon_output = aot_mod.get_output(0).numpy()

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


data_dtype = tvm.testing.parameter("int8", "uint8")
weight_dtype = tvm.testing.parameter("int8", "uint8")


@tvm.testing.requires_hexagon
def test_conv2d_relay_vrmpy(hexagon_session, data_dtype, weight_dtype):
    if data_dtype == "int8" and weight_dtype == "uint8":
        pytest.skip("(i8, u8) input pair is not supported")

    def get_conv2d_nchw(d_shape, w_shape, padding, strides=(1, 1)):
        out_dtype = "int32"

        data = relay.var("data", shape=d_shape, dtype=data_dtype)
        weight = relay.var("weight", shape=w_shape, dtype=weight_dtype)
        out_channel = w_shape[0]
        return relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=w_shape[2:],
            channels=out_channel,
            padding=padding,
            strides=strides,
            out_dtype=out_dtype,
        )

    target = get_hexagon_target("v68")
    I, O, H, W = 64, 256, 56, 56
    kH = kW = 3
    padding = (1, 1)
    strides = (1, 1)

    data_shape = (1, I, H, W)
    weight_shape = (O, I, kH, kW)
    bias_shape = (weight_shape[0],)

    bias = relay.var("bias", shape=bias_shape, dtype="int32")

    conv2d = get_conv2d_nchw(
        data_shape,
        weight_shape,
        padding,
        strides=strides,
    )
    bias_add = relay.nn.bias_add(conv2d, bias)
    mod = tvm.IRModule.from_expr(bias_add)

    if data_dtype == "uint8":
        data_np = np.random.uniform(0, 255, size=data_shape).astype("uint8")
    else:
        data_np = np.random.uniform(-128, 127, size=data_shape).astype("int8")

    if weight_dtype == "uint8":
        weight_np = np.random.uniform(0, 255, size=weight_shape).astype("uint8")
    else:
        weight_np = np.random.uniform(-128, 127, size=weight_shape).astype("int8")

    bias_np = np.random.randint(low=-127, high=128, size=bias_shape).astype("int32")
    params = {"weight": weight_np, "bias": bias_np}

    ref = (
        relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm")
        .evaluate()(*[data_np, weight_np, bias_np])
        .numpy()
    )

    with tvm.transform.PassContext(
        opt_level=3,
    ):
        executor = relay.backend.Executor("graph", {"link-params": True})
        lib = relay.build(mod, target=target, params=params, executor=executor)

    asm = lib.lib.get_source("asm")
    assert "vrmpy" in asm

    rt_mod = hexagon_session.get_executor_from_factory(lib)

    rt_mod.set_input("data", data_np)

    rt_mod.run()

    out = rt_mod.get_output(0).numpy()

    np.testing.assert_equal(out, ref)


@tvm.testing.requires_hexagon
def test_dense_relay_vrmpy(hexagon_session, data_dtype, weight_dtype):
    if data_dtype == "int8" and weight_dtype == "uint8":
        pytest.skip("(i8, u8) input pair is not supported")

    target = get_hexagon_target("v68")

    M = 128
    N = 1000
    K = 2048
    data_shape = (M, K)
    weight_shape = (N, K)

    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=weight_dtype)

    dense = relay.nn.dense(data, weight, out_dtype="int32")

    if data_dtype == "uint8":
        data_np = np.random.uniform(0, 255, size=data_shape).astype("uint8")
    else:
        data_np = np.random.uniform(-128, 127, size=data_shape).astype("int8")

    if weight_dtype == "uint8":
        weight_np = np.random.uniform(0, 255, size=weight_shape).astype("uint8")
    else:
        weight_np = np.random.uniform(-128, 127, size=weight_shape).astype("int8")

    bias_np = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

    params = {"weight": weight_np, "bias": bias_np}

    bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
    bias_add = relay.nn.bias_add(dense, bias)
    mod = tvm.IRModule.from_expr(bias_add)

    with tvm.transform.PassContext(
        opt_level=3,
    ):
        executor = relay.backend.Executor("graph", {"link-params": True})
        lib = relay.build(mod, target=target, params=params, executor=executor)

    asm = lib.lib.get_source("asm")
    assert "vrmpy" in asm

    rt_mod = hexagon_session.get_executor_from_factory(lib)

    rt_mod.set_input("data", data_np)

    rt_mod.run()

    out = rt_mod.get_output(0).numpy()

    ref = np.dot(data_np.astype("int32"), weight_np.transpose().astype("int32"))
    ref += bias_np

    np.testing.assert_equal(out, ref)


@tvm.testing.requires_hexagon
def test_lwp(
    hexagon_server_process,
    hexagon_launcher: HexagonLauncherRPC,
    hexagon_session: Session,
    hexagon_debug,
):
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

    weight_in = np.random.rand(5, 5, 3, 8).astype(dtype=dtype)
    data_in = np.random.rand(1, 64, 64, 3).astype(dtype=dtype)
    params = {"weight": weight_in}
    inputs = {"data": data_in}

    with tvm.transform.PassContext(opt_level=3, config={"tir.instrument_lwp": True}):
        lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            runtime=runtime,
            executor=executor,
        )
        # Create HexagonProfiler object
        dso_binary = "test_binary.so"
        profiler = HexagonProfiler(dso_binary, lowered, hexagon_server_process, hexagon_debug)

    graph_mod = hexagon_session.get_executor_from_factory(lowered)
    graph_mod.set_input(**params)
    graph_mod.run(**inputs)
    hexagon_output = graph_mod.get_output(0).numpy()

    # Get lightweight profiling output as a CSV file
    profiler.get_profile_output(hexagon_launcher, hexagon_session)

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=runtime,
            executor=executor,
        )
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(weight=weight_in)
    llvm_graph_mod.run(data=data_in)
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


@tvm.testing.requires_hexagon
def test_lwp_multiple_conv2d(
    hexagon_server_process,
    hexagon_launcher: HexagonLauncherRPC,
    hexagon_session: Session,
    hexagon_debug,
):
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

    with tvm.transform.PassContext(opt_level=3, config={"tir.instrument_lwp": True}):
        lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            runtime=runtime,
            executor=executor,
        )
        # Create HexagonProfiler object
        dso_binary = "test_binary.so"
        profiler = HexagonProfiler(dso_binary, lowered, hexagon_server_process, hexagon_debug)

    graph_mod = hexagon_session.get_executor_from_factory(lowered)
    graph_mod.set_input(**params)
    graph_mod.run(**inputs)
    hexagon_output = graph_mod.get_output(0).numpy()

    # Get lightweight profiling output as a CSV file
    profiler.get_profile_output(hexagon_launcher, hexagon_session)

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


if __name__ == "__main__":
    tvm.testing.main()

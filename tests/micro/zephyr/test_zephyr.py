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

import contextlib
import copy
import datetime
import glob
import logging
import os
import pathlib
import subprocess
import sys
import logging

import pytest
import numpy as np
import onnx
from PIL import Image

import tvm
import tvm.rpc
import tvm.micro
import tvm.testing
import tvm.relay as relay
from tvm.relay.testing import byoc

from tvm.contrib import utils
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end

import conftest

_LOG = logging.getLogger(__name__)

PLATFORMS = conftest.PLATFORMS


def _make_sess_from_op(
    temp_dir, model, zephyr_board, west_cmd, op_name, sched, arg_bufs, build_config
):
    target = tvm.target.target.micro(model)
    target = tvm.target.Target(target=target, host=target)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.build(sched, arg_bufs, target=target, name=op_name)

    return _make_session(temp_dir, zephyr_board, west_cmd, mod, build_config)


TEMPLATE_PROJECT_DIR = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / ".."
    / "apps"
    / "microtvm"
    / "zephyr"
    / "template_project"
).resolve()


def _make_session(temp_dir, zephyr_board, west_cmd, mod, build_config):
    project = tvm.micro.generate_project(
        str(TEMPLATE_PROJECT_DIR),
        mod,
        temp_dir / "project",
        {
            "project_type": "host_driven",
            "west_cmd": west_cmd,
            "verbose": bool(build_config.get("debug")),
            "zephyr_board": zephyr_board,
        },
    )
    if not build_config.get("skip_build"):
        project.build()
    project.flash()
    return tvm.micro.Session(project.transport())


def _make_add_sess(temp_dir, model, zephyr_board, west_cmd, build_config, dtype="int8"):
    A = tvm.te.placeholder((2,), dtype=dtype)
    B = tvm.te.placeholder((1,), dtype=dtype)
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)
    return _make_sess_from_op(
        temp_dir, model, zephyr_board, west_cmd, "add", sched, [A, B, C], build_config
    )


# The same test code can be executed on both the QEMU simulation and on real hardware.
@tvm.testing.requires_micro
def test_add_uint(temp_dir, platform, west_cmd, skip_build, tvm_debug):
    """Test compiling the on-device runtime."""

    model, zephyr_board = PLATFORMS[platform]
    build_config = {"skip_build": skip_build, "debug": tvm_debug}

    # NOTE: run test in a nested function so cPython will delete arrays before closing the session.
    def test_basic_add(sess):
        A_data = tvm.nd.array(np.array([2, 3], dtype="int8"), device=sess.device)
        assert (A_data.numpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype="int8"), device=sess.device)
        assert (B_data.numpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype="int8"), device=sess.device)
        assert (C_data.numpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        system_lib.get_function("add")(A_data, B_data, C_data)
        assert (C_data.numpy() == np.array([6, 7])).all()

    with _make_add_sess(temp_dir, model, zephyr_board, west_cmd, build_config) as sess:
        test_basic_add(sess)


def has_fpu(zephyr_board):
    sys.path.insert(0, str(TEMPLATE_PROJECT_DIR))
    try:
        import microtvm_api_server
    finally:
        sys.path.pop(0)

    return microtvm_api_server.Handler._has_fpu(zephyr_board)


# The same test code can be executed on both the QEMU simulation and on real hardware.
@tvm.testing.requires_micro
def test_add_float(temp_dir, platform, west_cmd, skip_build, tvm_debug):
    """Test compiling the on-device runtime."""
    model, zephyr_board = PLATFORMS[platform]
    if not has_fpu(zephyr_board):
        pytest.skip(f"FPU not enabled for {platform}")

    build_config = {"skip_build": skip_build, "debug": tvm_debug}

    # NOTE: run test in a nested function so cPython will delete arrays before closing the session.
    def test_basic_add(sess):
        A_data = tvm.nd.array(np.array([2.5, 3.5], dtype="float32"), device=sess.device)
        assert (A_data.numpy() == np.array([2.5, 3.5])).all()
        B_data = tvm.nd.array(np.array([4.5], dtype="float32"), device=sess.device)
        assert (B_data.numpy() == np.array([4.5])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype="float32"), device=sess.device)
        assert (C_data.numpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        system_lib.get_function("add")(A_data, B_data, C_data)
        assert (C_data.numpy() == np.array([7, 8])).all()

    with _make_add_sess(
        temp_dir, model, zephyr_board, west_cmd, build_config, dtype="float32"
    ) as sess:
        test_basic_add(sess)


@tvm.testing.requires_micro
def test_platform_timer(temp_dir, platform, west_cmd, skip_build, tvm_debug):
    """Test compiling the on-device runtime."""

    model, zephyr_board = PLATFORMS[platform]
    build_config = {"skip_build": skip_build, "debug": tvm_debug}

    # NOTE: run test in a nested function so cPython will delete arrays before closing the session.
    def test_basic_add(sess):
        A_data = tvm.nd.array(np.array([2, 3], dtype="int8"), device=sess.device)
        assert (A_data.numpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype="int8"), device=sess.device)
        assert (B_data.numpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype="int8"), device=sess.device)
        assert (C_data.numpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        time_eval_f = system_lib.time_evaluator(
            "add", sess.device, number=20, repeat=3, min_repeat_ms=40
        )
        result = time_eval_f(A_data, B_data, C_data)
        assert (C_data.numpy() == np.array([6, 7])).all()
        assert result.mean > 0
        assert len(result.results) == 3

    with _make_add_sess(temp_dir, model, zephyr_board, west_cmd, build_config) as sess:
        test_basic_add(sess)


@tvm.testing.requires_micro
def test_relay(temp_dir, platform, west_cmd, skip_build, tvm_debug):
    """Testing a simple relay graph"""
    model, zephyr_board = PLATFORMS[platform]
    build_config = {"skip_build": skip_build, "debug": tvm_debug}
    shape = (10,)
    dtype = "int8"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)

    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(func, target=target)

    with _make_session(
        temp_dir, zephyr_board, west_cmd, mod, build_config
    ) as session:
        graph_mod = tvm.micro.create_local_graph_executor(
            mod.get_graph_json(), session.get_system_lib(), session.device
        )
        graph_mod.set_input(**mod.get_params())
        x_in = np.random.randint(10, size=shape[0], dtype=dtype)
        graph_mod.run(x=x_in)
        result = graph_mod.get_output(0).numpy()
        tvm.testing.assert_allclose(graph_mod.get_input(0).numpy(), x_in)
        tvm.testing.assert_allclose(result, x_in * x_in + 1)


@tvm.testing.requires_micro
def test_onnx(temp_dir, platform, west_cmd, skip_build, tvm_debug):
    """Testing a simple ONNX model."""
    model, zephyr_board = PLATFORMS[platform]
    build_config = {"skip_build": skip_build, "debug": tvm_debug}

    # Load test images.
    this_dir = os.path.dirname(__file__)
    digit_2 = Image.open(f"{this_dir}/testdata/digit-2.jpg").resize((28, 28))
    digit_2 = np.asarray(digit_2).astype("float32")
    digit_2 = np.expand_dims(digit_2, axis=0)

    digit_9 = Image.open(f"{this_dir}/testdata/digit-9.jpg").resize((28, 28))
    digit_9 = np.asarray(digit_9).astype("float32")
    digit_9 = np.expand_dims(digit_9, axis=0)

    # Load ONNX model and convert to Relay.
    onnx_model = onnx.load(f"{this_dir}/testdata/mnist-8.onnx")
    shape = {"Input3": (1, 1, 28, 28)}
    relay_mod, params = relay.frontend.from_onnx(onnx_model, shape=shape, freeze_params=True)
    relay_mod = relay.transform.DynamicToStatic()(relay_mod)

    # We add the -link-params=1 option to ensure the model parameters are compiled in.
    # There is currently a bug preventing the host_driven environment from receiving
    # the model weights when set using graph_mod.set_input().
    # See: https://github.com/apache/tvm/issues/7567
    target = tvm.target.target.micro(model, options=["-link-params=1"])
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params)
        graph = lowered.get_graph_json()

    with _make_session(
        temp_dir, zephyr_board, west_cmd, lowered, build_config
    ) as session:
        graph_mod = tvm.micro.create_local_graph_executor(
            graph, session.get_system_lib(), session.device
        )

        # Send the digit-2 image and confirm that the correct result is returned.
        graph_mod.set_input("Input3", tvm.nd.array(digit_2))
        graph_mod.run()
        result = graph_mod.get_output(0).numpy()
        assert np.argmax(result) == 2

        # Send the digit-9 image and confirm that the correct result is returned.
        graph_mod.set_input("Input3", tvm.nd.array(digit_9))
        graph_mod.run()
        result = graph_mod.get_output(0).numpy()
        assert np.argmax(result) == 9


def check_result(
    temp_dir, relay_mod, model, zephyr_board, west_cmd, map_inputs, out_shape, result, build_config
):
    """Helper function to verify results"""
    TOL = 1e-5
    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(relay_mod, target=target)

    with _make_session(
        temp_dir, zephyr_board, west_cmd, mod, build_config
    ) as session:
        rt_mod = tvm.micro.create_local_graph_executor(
            mod.get_graph_json(), session.get_system_lib(), session.device
        )
        rt_mod.set_input(**mod.get_params())
        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**mod.get_params())
        rt_mod.run()

        out_shapes = out_shape if isinstance(out_shape, list) else [out_shape]
        results = result if isinstance(result, list) else [result]

        for idx, shape in enumerate(out_shapes):
            out = tvm.nd.empty(shape, device=session.device)
            out = rt_mod.get_output(idx, out)
            tvm.testing.assert_allclose(out.numpy(), results[idx], rtol=TOL, atol=TOL)


@tvm.testing.requires_micro
def test_byoc_microtvm(temp_dir, platform, west_cmd, skip_build, tvm_debug):
    """This is a simple test case to check BYOC capabilities of microTVM"""
    model, zephyr_board = PLATFORMS[platform]
    build_config = {"skip_build": skip_build, "debug": tvm_debug}
    x = relay.var("x", shape=(10, 10))
    w0 = relay.var("w0", shape=(10, 10))
    w1 = relay.var("w1", shape=(10, 10))
    w2 = relay.var("w2", shape=(10, 10))
    w3 = relay.var("w3", shape=(10, 10))
    w4 = relay.var("w4", shape=(10, 10))
    w5 = relay.var("w5", shape=(10, 10))
    w6 = relay.var("w6", shape=(10, 10))
    w7 = relay.var("w7", shape=(10, 10))

    # C compiler
    z0 = relay.add(x, w0)
    p0 = relay.subtract(z0, w1)
    q0 = relay.multiply(p0, w2)

    z1 = relay.add(x, w3)
    p1 = relay.subtract(z1, w4)
    q1 = relay.multiply(p1, w5)

    # Other parts on TVM
    z2 = relay.add(x, w6)
    q2 = relay.subtract(z2, w7)

    r = relay.concatenate((q0, q1, q2), axis=0)
    f = relay.Function([x, w0, w1, w2, w3, w4, w5, w6, w7], r)
    mod = tvm.IRModule()
    ann = byoc.CcompilerAnnotator()
    mod["main"] = ann.visit(f)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    mod = tvm.relay.transform.InferType()(mod)

    x_data = np.random.rand(10, 10).astype("float32")
    w_data = []
    for _ in range(8):
        w_data.append(np.random.rand(10, 10).astype("float32"))

    map_inputs = {"w{}".format(i): w_data[i] for i in range(8)}
    map_inputs["x"] = x_data
    check_result(
        temp_dir=temp_dir,
        relay_mod=mod,
        map_inputs=map_inputs,
        out_shape=(30, 10),
        result=np.concatenate(
            (
                ((x_data + w_data[0]) - w_data[1]) * w_data[2],
                ((x_data + w_data[3]) - w_data[4]) * w_data[5],
                x_data + w_data[6] - w_data[7],
            ),
            axis=0,
        ),
        model=model,
        zephyr_board=zephyr_board,
        west_cmd=west_cmd,
        build_config=build_config,
    )


def _make_add_sess_with_shape(temp_dir, model, zephyr_board, west_cmd, shape, build_config):
    A = tvm.te.placeholder(shape, dtype="int8")
    C = tvm.te.compute(A.shape, lambda i: A[i] + A[i], name="C")
    sched = tvm.te.create_schedule(C.op)
    return _make_sess_from_op(
        temp_dir, model, zephyr_board, west_cmd, "add", sched, [A, C], build_config
    )


@pytest.mark.parametrize(
    "shape,",
    [
        pytest.param((1 * 1024,), id="(1*1024)"),
        pytest.param((4 * 1024,), id="(4*1024)"),
        pytest.param((16 * 1024,), id="(16*1024)"),
    ],
)
@tvm.testing.requires_micro
def test_rpc_large_array(temp_dir, platform, west_cmd, skip_build, tvm_debug, shape):
    """Test large RPC array transfer."""
    model, zephyr_board = PLATFORMS[platform]
    build_config = {"skip_build": skip_build, "debug": tvm_debug}

    # NOTE: run test in a nested function so cPython will delete arrays before closing the session.
    def test_tensors(sess):
        a_np = np.random.randint(low=-128, high=127, size=shape, dtype="int8")

        A_data = tvm.nd.array(a_np, device=sess.device)
        assert (A_data.numpy() == a_np).all()
        C_data = tvm.nd.array(np.zeros(shape, dtype="int8"), device=sess.device)
        assert (C_data.numpy() == np.zeros(shape)).all()

    with _make_add_sess_with_shape(
        temp_dir, model, zephyr_board, west_cmd, shape, build_config
    ) as sess:
        test_tensors(sess)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))

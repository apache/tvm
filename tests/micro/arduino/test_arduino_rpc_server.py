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

"""
This unit test simulates an autotuning workflow, where we:
1. Instantiate the Arduino RPC server project
2. Build and flash that project onto our target board

"""

import pathlib
import numpy as np
import onnx
import pytest

import tvm
import tvm.testing
from PIL import Image
from tvm import relay
from tvm.relay.testing import byoc
from tvm.relay.backend import Executor, Runtime

import test_utils


def _make_session(
    model,
    arduino_board,
    workspace_dir,
    mod,
    build_config,
    serial_number: str = None,
):
    project = tvm.micro.generate_project(
        str(test_utils.TEMPLATE_PROJECT_DIR),
        mod,
        workspace_dir / "project",
        {
            "board": arduino_board,
            "project_type": "host_driven",
            "verbose": bool(build_config.get("debug")),
            "serial_number": serial_number,
        },
    )
    project.build()
    project.flash()
    return tvm.micro.Session(project.transport())


def _make_sess_from_op(
    model,
    arduino_board,
    workspace_dir,
    op_name,
    sched,
    arg_bufs,
    build_config,
    serial_number: str = None,
):
    target = tvm.target.target.micro(model)
    runtime = Runtime("crt", {"system-lib": True})
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.build(sched, arg_bufs, target=target, runtime=runtime, name=op_name)

    return _make_session(model, arduino_board, workspace_dir, mod, build_config, serial_number)


def _make_add_sess(model, arduino_board, workspace_dir, build_config, serial_number: str = None):
    A = tvm.te.placeholder((2,), dtype="int8")
    B = tvm.te.placeholder((1,), dtype="int8")
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)
    return _make_sess_from_op(
        model,
        arduino_board,
        workspace_dir,
        "add",
        sched,
        [A, B, C],
        build_config,
        serial_number,
    )


# The same test code can be executed on both the QEMU simulation and on real hardware.
@tvm.testing.requires_micro
@pytest.mark.requires_hardware
def test_compile_runtime(board, microtvm_debug, workspace_dir, serial_number):
    """Test compiling the on-device runtime."""

    model = test_utils.ARDUINO_BOARDS[board]
    build_config = {"debug": microtvm_debug}

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

    with _make_add_sess(model, board, workspace_dir, build_config, serial_number) as sess:
        test_basic_add(sess)


@tvm.testing.requires_micro
@pytest.mark.requires_hardware
def test_platform_timer(board, microtvm_debug, workspace_dir, serial_number):
    """Test compiling the on-device runtime."""

    model = test_utils.ARDUINO_BOARDS[board]
    build_config = {"debug": microtvm_debug}

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

    with _make_add_sess(model, board, workspace_dir, build_config, serial_number) as sess:
        test_basic_add(sess)


@tvm.testing.requires_micro
@pytest.mark.requires_hardware
def test_relay(board, microtvm_debug, workspace_dir, serial_number):
    """Testing a simple relay graph"""
    model = test_utils.ARDUINO_BOARDS[board]
    build_config = {"debug": microtvm_debug}

    shape = (10,)
    dtype = "int8"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)

    target = tvm.target.target.micro(model)
    runtime = Runtime("crt", {"system-lib": True})
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(func, target=target, runtime=runtime)

    with _make_session(model, board, workspace_dir, mod, build_config, serial_number) as session:
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
@pytest.mark.requires_hardware
def test_onnx(board, microtvm_debug, workspace_dir, serial_number):
    """Testing a simple ONNX model."""
    model = test_utils.ARDUINO_BOARDS[board]
    build_config = {"debug": microtvm_debug}

    # Load test images.
    this_dir = pathlib.Path(__file__).parent
    mnist_testdata = this_dir.parent / "testdata" / "mnist"
    digit_2 = Image.open(mnist_testdata / "digit-2.jpg").resize((28, 28))
    digit_2 = np.asarray(digit_2).astype("float32")
    digit_2 = np.expand_dims(digit_2, axis=0)

    digit_9 = Image.open(mnist_testdata / "digit-9.jpg").resize((28, 28))
    digit_9 = np.asarray(digit_9).astype("float32")
    digit_9 = np.expand_dims(digit_9, axis=0)

    # Load ONNX model and convert to Relay.
    onnx_model = onnx.load(mnist_testdata / "mnist-8.onnx")
    shape = {"Input3": (1, 1, 28, 28)}
    relay_mod, params = relay.frontend.from_onnx(onnx_model, shape=shape, freeze_params=True)
    relay_mod = relay.transform.DynamicToStatic()(relay_mod)

    target = tvm.target.target.micro(model)
    runtime = Runtime("crt", {"system-lib": True})
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        executor = Executor("graph", {"link-params": True})
        lowered = relay.build(relay_mod, target, params=params, executor=executor, runtime=runtime)
        graph = lowered.get_graph_json()

    with _make_session(
        model, board, workspace_dir, lowered, build_config, serial_number
    ) as session:
        graph_mod = tvm.micro.create_local_graph_executor(
            graph, session.get_system_lib(), session.device
        )

        # Send the digit-2 image and confirm that the correct result is returned.
        graph_mod.set_input("Input3", tvm.nd.array(digit_2))
        graph_mod.run()
        result = graph_mod.get_output(0).numpy()
        print(result)
        assert np.argmax(result) == 2

        # Send the digit-9 image and confirm that the correct result is returned.
        graph_mod.set_input("Input3", tvm.nd.array(digit_9))
        graph_mod.run()
        result = graph_mod.get_output(0).numpy()
        assert np.argmax(result) == 9


def check_result(
    relay_mod,
    model,
    arduino_board,
    workspace_dir,
    map_inputs,
    out_shape,
    result,
    build_config,
    serial_number,
):
    """Helper function to verify results"""
    TOL = 1e-5
    target = tvm.target.target.micro(model)
    runtime = Runtime("crt", {"system-lib": True})
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(relay_mod, target=target, runtime=runtime)

    with _make_session(
        model, arduino_board, workspace_dir, mod, build_config, serial_number
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
@pytest.mark.requires_hardware
def test_byoc_microtvm(board, microtvm_debug, workspace_dir, serial_number):
    """This is a simple test case to check BYOC capabilities of microTVM"""
    model = test_utils.ARDUINO_BOARDS[board]
    build_config = {"debug": microtvm_debug}

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
        build_config=build_config,
        arduino_board=board,
        workspace_dir=workspace_dir,
        serial_number=serial_number,
    )


def _make_add_sess_with_shape(
    model,
    arduino_board,
    workspace_dir,
    shape,
    build_config,
    serial_number: str = None,
):
    A = tvm.te.placeholder(shape, dtype="int8")
    C = tvm.te.compute(A.shape, lambda i: A[i] + A[i], name="C")
    sched = tvm.te.create_schedule(C.op)
    return _make_sess_from_op(
        model,
        arduino_board,
        workspace_dir,
        "add",
        sched,
        [A, C],
        build_config,
        serial_number,
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
@pytest.mark.requires_hardware
def test_rpc_large_array(board, microtvm_debug, workspace_dir, shape, serial_number):
    """Test large RPC array transfer."""
    model = test_utils.ARDUINO_BOARDS[board]
    build_config = {"debug": microtvm_debug}

    # NOTE: run test in a nested function so cPython will delete arrays before closing the session.
    def test_tensors(sess):
        a_np = np.random.randint(low=-128, high=127, size=shape, dtype="int8")

        A_data = tvm.nd.array(a_np, device=sess.device)
        assert (A_data.numpy() == a_np).all()
        C_data = tvm.nd.array(np.zeros(shape, dtype="int8"), device=sess.device)
        assert (C_data.numpy() == np.zeros(shape)).all()

    with _make_add_sess_with_shape(
        model, board, workspace_dir, shape, build_config, serial_number
    ) as sess:
        test_tensors(sess)


if __name__ == "__main__":
    tvm.testing.main()

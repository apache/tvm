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
import logging
import os
import pathlib
import logging

import pytest
import numpy as np

import onnx
from PIL import Image

import tvm
import tvm.testing
import tvm.relay as relay
from tvm.relay.backend import Executor, Runtime
from tvm.relay.testing import byoc
from tvm.contrib import utils
from tvm.micro.testing.utils import check_tune_log

import test_utils

_LOG = logging.getLogger(__name__)


def _make_sess_from_op(
    temp_dir, model, zephyr_board, west_cmd, op_name, sched, arg_bufs, build_config, use_fvp
):
    runtime = Runtime("crt", {"system-lib": True})
    target = tvm.target.target.micro(model)
    target = tvm.target.Target(target=target, host=target)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.build(sched, arg_bufs, target=target, runtime=runtime, name=op_name)

    return _make_session(temp_dir, zephyr_board, west_cmd, mod, build_config, use_fvp)


def _make_session(temp_dir, zephyr_board, west_cmd, mod, build_config, use_fvp):
    config_main_stack_size = None
    if test_utils.qemu_boards(zephyr_board):
        config_main_stack_size = 1536

    project_options = {
        "project_type": "host_driven",
        "west_cmd": west_cmd,
        "verbose": bool(build_config.get("debug")),
        "board": zephyr_board,
        "arm_fvp_path": "/opt/arm/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55",
        "use_fvp": bool(use_fvp),
    }
    if config_main_stack_size is not None:
        project_options["config_main_stack_size"] = config_main_stack_size

    project = tvm.micro.generate_project(
        str(test_utils.TEMPLATE_PROJECT_DIR),
        mod,
        temp_dir / "project",
        project_options,
    )
    project.build()
    project.flash()
    return tvm.micro.Session(project.transport())


def _make_add_sess(temp_dir, model, zephyr_board, west_cmd, build_config, use_fvp, dtype="int8"):
    A = tvm.te.placeholder((2,), dtype=dtype)
    B = tvm.te.placeholder((1,), dtype=dtype)
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)
    return _make_sess_from_op(
        temp_dir, model, zephyr_board, west_cmd, "add", sched, [A, B, C], build_config, use_fvp
    )


# The same test code can be executed on both the QEMU simulation and on real hardware.
@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521"])
@pytest.mark.xfail_on_fvp()
def test_add_uint(workspace_dir, board, west_cmd, microtvm_debug, use_fvp):
    """Test compiling the on-device runtime."""

    model = test_utils.ZEPHYR_BOARDS[board]
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

    with _make_add_sess(workspace_dir, model, board, west_cmd, build_config, use_fvp) as sess:
        test_basic_add(sess)


# The same test code can be executed on both the QEMU simulation and on real hardware.
@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521"])
@pytest.mark.xfail_on_fvp()
def test_add_float(workspace_dir, board, west_cmd, microtvm_debug, use_fvp):
    """Test compiling the on-device runtime."""
    model = test_utils.ZEPHYR_BOARDS[board]
    if not test_utils.has_fpu(board):
        pytest.skip(f"FPU not enabled for {board}")

    build_config = {"debug": microtvm_debug}

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
        workspace_dir, model, board, west_cmd, build_config, use_fvp, dtype="float32"
    ) as sess:
        test_basic_add(sess)


@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521"])
@pytest.mark.xfail_on_fvp()
def test_platform_timer(workspace_dir, board, west_cmd, microtvm_debug, use_fvp):
    """Test compiling the on-device runtime."""

    model = test_utils.ZEPHYR_BOARDS[board]
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

    with _make_add_sess(workspace_dir, model, board, west_cmd, build_config, use_fvp) as sess:
        test_basic_add(sess)


@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521"])
@pytest.mark.xfail_on_fvp()
def test_relay(workspace_dir, board, west_cmd, microtvm_debug, use_fvp):
    """Testing a simple relay graph"""
    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": microtvm_debug}
    shape = (10,)
    dtype = "int8"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)
    ir_mod = tvm.IRModule.from_expr(func)

    runtime = Runtime("crt", {"system-lib": True})
    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(ir_mod, target=target, runtime=runtime)

    with _make_session(workspace_dir, board, west_cmd, mod, build_config, use_fvp) as session:
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
@pytest.mark.skip_boards(["mps2_an521"])
@pytest.mark.xfail_on_fvp()
def test_onnx(workspace_dir, board, west_cmd, microtvm_debug, use_fvp):
    """Testing a simple ONNX model."""
    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": microtvm_debug}

    this_dir = pathlib.Path(os.path.dirname(__file__))
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

    # We add the link-params=True option to ensure the model parameters are compiled in.
    # There is currently a bug preventing the host_driven environment from receiving
    # the model weights when set using graph_mod.set_input().
    # See: https://github.com/apache/tvm/issues/7567
    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        executor = Executor("graph", {"link-params": True})
        runtime = Runtime("crt", {"system-lib": True})
        lowered = relay.build(relay_mod, target, params=params, executor=executor, runtime=runtime)
        graph = lowered.get_graph_json()

    with _make_session(workspace_dir, board, west_cmd, lowered, build_config, use_fvp) as session:
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
    temp_dir,
    relay_mod,
    model,
    zephyr_board,
    west_cmd,
    map_inputs,
    out_shape,
    result,
    build_config,
    use_fvp,
):
    """Helper function to verify results"""
    TOL = 1e-5
    runtime = Runtime("crt", {"system-lib": True})
    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(relay_mod, target=target, runtime=runtime)

    with _make_session(temp_dir, zephyr_board, west_cmd, mod, build_config, use_fvp) as session:
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
@pytest.mark.skip_boards(["mps2_an521"])
@pytest.mark.xfail_on_fvp()
def test_byoc_microtvm(workspace_dir, board, west_cmd, microtvm_debug, use_fvp):
    """This is a simple test case to check BYOC capabilities of microTVM"""
    model = test_utils.ZEPHYR_BOARDS[board]
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
        temp_dir=workspace_dir,
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
        zephyr_board=board,
        west_cmd=west_cmd,
        build_config=build_config,
        use_fvp=use_fvp,
    )


def _make_add_sess_with_shape(
    temp_dir, model, zephyr_board, west_cmd, shape, build_config, use_fvp
):
    A = tvm.te.placeholder(shape, dtype="int8")
    C = tvm.te.compute(A.shape, lambda i: A[i] + A[i], name="C")
    sched = tvm.te.create_schedule(C.op)
    return _make_sess_from_op(
        temp_dir, model, zephyr_board, west_cmd, "add", sched, [A, C], build_config, use_fvp
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
@pytest.mark.skip_boards(["mps2_an521"])
@pytest.mark.xfail_on_fvp()
def test_rpc_large_array(workspace_dir, board, west_cmd, microtvm_debug, shape, use_fvp):
    """Test large RPC array transfer."""
    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": microtvm_debug}

    # NOTE: run test in a nested function so cPython will delete arrays before closing the session.
    def test_tensors(sess):
        a_np = np.random.randint(low=-128, high=127, size=shape, dtype="int8")

        A_data = tvm.nd.array(a_np, device=sess.device)
        assert (A_data.numpy() == a_np).all()
        C_data = tvm.nd.array(np.zeros(shape, dtype="int8"), device=sess.device)
        assert (C_data.numpy() == np.zeros(shape)).all()

    with _make_add_sess_with_shape(
        workspace_dir, model, board, west_cmd, shape, build_config, use_fvp
    ) as sess:
        test_tensors(sess)


@pytest.mark.xfail(strict=False, reason="See https://github.com/apache/tvm/issues/10297")
@tvm.testing.requires_micro
def test_autotune_conv2d(workspace_dir, board, west_cmd, microtvm_debug, use_fvp):
    """Test AutoTune for microTVM Zephyr"""
    if board != "qemu_x86":
        pytest.xfail(f"Autotune fails on {board}.")

    runtime = Runtime("crt", {"system-lib": True})
    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": microtvm_debug}

    # Create a Relay model
    data_shape = (1, 3, 16, 16)
    weight_shape = (8, 3, 5, 5)
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)

    data_sample = np.random.rand(data_shape[0], data_shape[1], data_shape[2], data_shape[3]).astype(
        "float32"
    )
    weight_sample = np.random.rand(
        weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
    ).astype("float32")
    params = {mod["main"].params[1].name_hint: weight_sample}

    target = tvm.target.target.micro(model)
    pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
    with pass_context:
        tasks = tvm.autotvm.task.extract_from_program(mod["main"], {}, target)
    assert len(tasks) > 0

    config_main_stack_size = None
    if test_utils.qemu_boards(board):
        config_main_stack_size = 1536

    project_options = {
        "board": board,
        "west_cmd": west_cmd,
        "verbose": 1,
        "project_type": "host_driven",
        "use_fvp": bool(use_fvp),
    }
    if config_main_stack_size is not None:
        project_options["config_main_stack_size"] = config_main_stack_size

    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=test_utils.TEMPLATE_PROJECT_DIR,
        project_options=project_options,
    )

    timeout = 200
    builder = tvm.autotvm.LocalBuilder(
        timeout=timeout,
        n_parallel=1,
        build_kwargs={"build_option": {"tir.disable_vectorize": True}},
        do_fork=True,
        build_func=tvm.micro.autotvm_build_func,
        runtime=runtime,
    )
    runner = tvm.autotvm.LocalRunner(
        number=1, repeat=1, timeout=timeout, module_loader=module_loader
    )

    measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

    log_path = pathlib.Path("zephyr_autotune.log")
    if log_path.exists():
        log_path.unlink()

    n_trial = 10
    for task in tasks:
        tuner = tvm.autotvm.tuner.GATuner(task)
        tuner.tune(
            n_trial=n_trial,
            measure_option=measure_option,
            callbacks=[
                tvm.autotvm.callback.log_to_file(str(log_path)),
                tvm.autotvm.callback.progress_bar(n_trial, si_prefix="M"),
            ],
            si_prefix="M",
        )
        assert tuner.best_flops > 0

    check_tune_log(log_path)

    # Build without tuning
    with pass_context:
        lowered = tvm.relay.build(mod, target=target, runtime=runtime, params=params)

    temp_dir = utils.tempdir()
    with _make_session(temp_dir, board, west_cmd, lowered, build_config, use_fvp) as session:
        graph_mod = tvm.micro.create_local_graph_executor(
            lowered.get_graph_json(), session.get_system_lib(), session.device
        )
        graph_mod.set_input(**lowered.get_params())
        graph_mod.run(data=data_sample)
        expected_output = graph_mod.get_output(0).numpy()
        del graph_mod

    # Build using autotune logs
    with tvm.autotvm.apply_history_best(str(log_path)):
        with pass_context:
            lowered_tuned = tvm.relay.build(mod, target=target, runtime=runtime, params=params)

    temp_dir = utils.tempdir()
    with _make_session(temp_dir, board, west_cmd, lowered_tuned, build_config, use_fvp) as session:
        graph_mod = tvm.micro.create_local_graph_executor(
            lowered_tuned.get_graph_json(), session.get_system_lib(), session.device
        )
        graph_mod.set_input(**lowered_tuned.get_params())
        graph_mod.run(data=data_sample)
        output = graph_mod.get_output(0).numpy()
        del graph_mod

    tvm.testing.assert_allclose(output, expected_output, rtol=1e-4, atol=1e-5)


@tvm.testing.requires_micro
def test_schedule_build_with_cmsis_dependency(
    workspace_dir, board, west_cmd, microtvm_debug, use_fvp
):
    """Test Relay schedule with CMSIS dependency. This test shows if microTVM Auto tuning
    with Zephyr breaks if CMSIS dependency was required for a schedule.
    """
    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": microtvm_debug}
    target = tvm.target.target.micro(model, options=["-keys=arm_cpu,cpu"])

    if not target.features.has_dsp:
        pytest.skip(f"ISA does not support DSP. target: {target}")

    # Create a Relay conv2d
    data_shape = (1, 16, 16, 3)
    weight_shape = (5, 5, 8, 3)
    data = relay.var("data", relay.TensorType(data_shape, "int8"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "int8"))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWOI",
        out_dtype="int32",
    )
    func = relay.Function([data, weight], y)
    ir_mod = tvm.IRModule.from_expr(func)

    runtime = Runtime("crt", {"system-lib": True})

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(ir_mod, target=target, runtime=runtime)

    project_options = {
        "project_type": "host_driven",
        "west_cmd": west_cmd,
        "verbose": bool(build_config.get("debug")),
        "board": board,
        "cmsis_path": os.getenv("CMSIS_PATH"),
        "use_fvp": bool(use_fvp),
    }

    project_dir = workspace_dir / "project"
    project = tvm.micro.generate_project(
        str(test_utils.TEMPLATE_PROJECT_DIR),
        mod,
        project_dir,
        project_options,
    )
    project.build()

    with open(project_dir / "CMakeLists.txt", "r") as cmake_f:
        cmake_content = cmake_f.read()

    assert "CMSIS/DSP/Include" in cmake_content
    assert "CMSIS/DSP/Include/dsp" in cmake_content
    assert "CMSIS/DSP/Include" in cmake_content
    assert "CMSIS/NN/Include" in cmake_content


if __name__ == "__main__":
    tvm.testing.main()

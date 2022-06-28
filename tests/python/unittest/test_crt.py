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
import os
import pathlib
import shutil
import pytest

pytest.importorskip("pty")

import pytest

import tvm
import tvm.relay
import tvm.testing
from tvm.target import Target
from tvm.relay.backend import Runtime
from tvm.relay.backend import Executor

BUILD = True
DEBUG = False

TARGET = tvm.target.target.micro("host")


def _make_sess_from_op(temp_dir, op_name, sched, arg_bufs):
    runtime = Runtime("crt", {"system-lib": True})
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.build(sched, arg_bufs, Target(TARGET, TARGET), runtime=runtime, name=op_name)

    return _make_session(temp_dir, mod)


def _make_session(temp_dir, mod):
    template_project_dir = os.path.join(tvm.micro.get_standalone_crt_dir(), "template", "host")
    project = tvm.micro.generate_project(
        template_project_dir, mod, temp_dir / "project", {"verbose": 1}
    )
    project.build()
    project.flash()
    return tvm.micro.Session(project.transport())


def _make_add_sess(temp_dir):
    A = tvm.te.placeholder((2,), dtype="int8")
    B = tvm.te.placeholder((1,), dtype="int8")
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)
    return _make_sess_from_op(temp_dir, "add", sched, [A, B, C])


def _make_ident_sess(temp_dir):
    A = tvm.te.placeholder((2,), dtype="int8")
    B = tvm.te.compute(A.shape, lambda i: A[i], name="B")
    sched = tvm.te.create_schedule(B.op)
    return _make_sess_from_op(temp_dir, "ident", sched, [A, B])


@tvm.testing.requires_micro
def test_compile_runtime():
    """Test compiling the on-device runtime."""
    import tvm.micro

    temp_dir = tvm.contrib.utils.tempdir()

    with _make_add_sess(temp_dir) as sess:
        A_data = tvm.nd.array(np.array([2, 3], dtype="int8"), device=sess.device)
        assert (A_data.numpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype="int8"), device=sess.device)
        assert (B_data.numpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype="int8"), device=sess.device)
        assert (C_data.numpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        system_lib.get_function("add")(A_data, B_data, C_data)
        assert (C_data.numpy() == np.array([6, 7])).all()


@tvm.testing.requires_micro
def test_compile_runtime_llvm():
    """Test targeting the on-device runtime with the llvm backend."""
    global TARGET
    old_target = TARGET
    try:
        # NOTE: test_compile_runtime uses the "c" backend--re run it using the llvm backend.
        target_str = str(TARGET)
        assert target_str.startswith("c ")
        TARGET = tvm.target.Target("llvm " + str(TARGET)[len("c ") :])

        test_compile_runtime()

    finally:
        TARGET = old_target


@tvm.testing.requires_micro
def test_reset():
    """Test when the remote end resets during a session."""
    import tvm.micro
    from tvm.micro import transport

    temp_dir = tvm.contrib.utils.tempdir()

    with _make_add_sess(temp_dir) as sess:
        try:
            sess._rpc.get_function("tvm.testing.reset_server")()
            assert False, "expected to raise SessionTerminatedError; did not raise"
        except tvm.micro.SessionTerminatedError:
            pass


@tvm.testing.requires_micro
def test_graph_executor():
    """Test use of the graph executor with microTVM."""

    ws_root = pathlib.Path(os.path.dirname(__file__) + "/micro-workspace")
    if ws_root.exists():
        shutil.rmtree(ws_root)
    temp_dir = tvm.contrib.utils.tempdir(ws_root.resolve())
    relay_mod = tvm.parser.fromtext(
        """
      #[version = "0.0.5"]
      def @main(%a : Tensor[(1, 2), uint8], %b : Tensor[(1, 2), uint8]) {
          %0 = %a + %b;
          %0
      }"""
    )

    runtime = Runtime("crt", {"system-lib": True})
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory = tvm.relay.build(relay_mod, target=TARGET, runtime=runtime)

    def do_test(graph_mod):

        A_data = tvm.nd.array(np.array([2, 3], dtype="uint8"), device=sess.device)
        assert (A_data.numpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4, 7], dtype="uint8"), device=sess.device)
        assert (B_data.numpy() == np.array([4, 7])).all()

        assert graph_mod.get_input_index("a") == 0
        assert graph_mod.get_input_index("b") == 1

        graph_mod.run(a=A_data, b=B_data)

        out = graph_mod.get_output(0)
        assert (out.numpy() == np.array([6, 10])).all()

    with _make_session(temp_dir, factory) as sess:

        graph_mod_local = tvm.micro.create_local_graph_executor(
            factory.get_graph_json(), sess.get_system_lib(), sess.device
        )

        do_test(graph_mod_local)

        graph_mod = tvm.contrib.graph_executor.create(
            factory.get_graph_json(), sess.get_system_lib(), sess.device
        )

        do_test(graph_mod)


@tvm.testing.requires_micro
def test_aot_executor():
    """Test use of the AOT executor with microTVM."""

    ws_root = pathlib.Path(os.path.dirname(__file__) + "/micro-workspace")
    if ws_root.exists():
        shutil.rmtree(ws_root)
    temp_dir = tvm.contrib.utils.tempdir(ws_root.resolve())
    relay_mod = tvm.parser.fromtext(
        """
      #[version = "0.0.5"]
      def @main(%a : Tensor[(1, 2), uint8], %b : Tensor[(1, 2), uint8]) {
          %0 = %a + %b;
          %0
      }"""
    )

    runtime = Runtime("crt", {"system-lib": True})
    executor = Executor("aot")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory = tvm.relay.build(relay_mod, target=TARGET, runtime=runtime, executor=executor)

    def do_test():
        aot_executor = tvm.runtime.executor.aot_executor.AotModule(
            sess._rpc.get_function("tvm.aot_executor.create")(
                sess.get_system_lib(), sess.device, "default"
            )
        )

        assert aot_executor.get_input_index("a") == 0
        assert aot_executor.get_input_index("b") == 1

        assert aot_executor.get_num_inputs() == 2
        assert aot_executor.get_num_outputs() == 1

        A_np = np.array([[2, 3]], dtype="uint8")
        B_np = np.array([[4, 7]], dtype="uint8")

        A_data = aot_executor.get_input("a").copyfrom(A_np)
        B_data = aot_executor.get_input("b").copyfrom(B_np)

        aot_executor.run()

        out = aot_executor.get_output(0)
        assert (out.numpy() == np.array([6, 10])).all()

        B_np_new = np.array([[5, 8]])
        aot_executor.set_input("b", B_np_new)
        assert (B_data.numpy() == B_np_new).all()

    with _make_session(temp_dir, factory) as sess:
        do_test()


enable_usmp, expect_exception = tvm.testing.parameters((True, True), (False, False))


@tvm.testing.requires_micro
def test_aot_executor_usmp_const_pool(enable_usmp, expect_exception):
    """Test the AOT executor with microTVM using usmp.
    Test should fail if const pool is supplied to executor
    as these are currently not supported
    """
    ws_root = pathlib.Path(os.path.dirname(__file__) + "/micro-workspace-usmp")
    if ws_root.exists():
        shutil.rmtree(ws_root)
    temp_dir = tvm.contrib.utils.tempdir(ws_root.resolve())
    relay_mod = tvm.parser.fromtext(
        """
      #[version = "0.0.5"]
      def @main(%a : Tensor[(1, 2), uint8], %b : Tensor[(1, 2), uint8], %c : Tensor[(1,2), uint8]) {
          %0 = %a + %b;
          %1 = %0 + %c;
          %1
      }"""
    )

    runtime = Runtime("crt", {"system-lib": True})
    executor = Executor("aot")
    main_func = relay_mod["main"]
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}
    B_np = np.array([[4, 7]], dtype="uint8").astype(type_dict["b"])
    C_np = np.array([[8, 9]], dtype="uint8").astype(type_dict["c"])
    params = {"c": C_np}
    with tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True, "tir.usmp.enable": enable_usmp}
    ):
        factory = tvm.relay.build(
            relay_mod,
            target=TARGET,
            runtime=runtime,
            executor=executor,
            params=params,
        )

    def do_test():
        try:
            aot_executor = tvm.runtime.executor.aot_executor.AotModule(
                sess._rpc.get_function("tvm.aot_executor.create")(
                    sess.get_system_lib(), sess.device, "default"
                )
            )
        except tvm._ffi.base.TVMError as e:
            if expect_exception:
                return
            else:
                raise e

        assert aot_executor.get_input_index("a") == 0
        assert aot_executor.get_input_index("b") == 1

        assert aot_executor.get_num_inputs() == 2
        assert aot_executor.get_num_outputs() == 1

        A_np = np.array([[2, 3]], dtype="uint8")
        B_np = np.array([[4, 7]], dtype="uint8")

        A_data = aot_executor.get_input("a").copyfrom(A_np)
        B_data = aot_executor.get_input("b").copyfrom(B_np)
        aot_executor.run()

        out = aot_executor.get_output(0)
        assert (out.numpy() == np.array([14, 19])).all()

        B_np_new = np.array([[5, 8]])
        aot_executor.set_input("b", B_np_new)
        assert (B_data.numpy() == B_np_new).all()

    with _make_session(temp_dir, factory) as sess:
        do_test()


@tvm.testing.requires_micro
def test_std_math_functions():
    """Verify that standard math functions can be used."""
    import tvm.micro

    temp_dir = tvm.contrib.utils.tempdir()

    with _make_add_sess(temp_dir) as sess:
        A_data = tvm.nd.array(np.array([2, 3], dtype="int8"), device=sess.device)
        assert (A_data.numpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype="int8"), device=sess.device)
        assert (B_data.numpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype="int8"), device=sess.device)
        assert (C_data.numpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        system_lib.get_function("add")(A_data, B_data, C_data)

    temp_dir = tvm.contrib.utils.tempdir()
    A = tvm.te.placeholder((2,), dtype="float32", name="A")
    B = tvm.te.compute(A.shape, lambda i: tvm.te.exp(A[i]), name="B")
    s = tvm.te.create_schedule(B.op)

    with _make_sess_from_op(temp_dir, "myexpf", s, [A, B]) as sess:
        A_data = tvm.nd.array(np.array([2.0, 3.0], dtype="float32"), device=sess.device)
        B_data = tvm.nd.array(np.array([2.0, 3.0], dtype="float32"), device=sess.device)
        lib = sess.get_system_lib()
        func = lib["myexpf"]
        func(A_data, B_data)
        np.testing.assert_allclose(B_data.numpy(), np.array([7.389056, 20.085537]))


@tvm.testing.requires_micro
def test_platform_timer():
    """Verify the platform timer can be used to time remote functions."""
    import tvm.micro

    temp_dir = tvm.contrib.utils.tempdir()
    A = tvm.te.placeholder((2,), dtype="float32", name="A")
    B = tvm.te.compute(A.shape, lambda i: tvm.te.exp(A[i]), name="B")
    s = tvm.te.create_schedule(B.op)

    with _make_sess_from_op(temp_dir, "myexpf", s, [A, B]) as sess:
        A_data = tvm.nd.array(np.array([2.0, 3.0], dtype="float32"), device=sess.device)
        B_data = tvm.nd.array(np.array([2.0, 3.0], dtype="float32"), device=sess.device)
        lib = sess.get_system_lib()
        time_eval_f = lib.time_evaluator(
            "myexpf", sess.device, number=2000, repeat=3, min_repeat_ms=40
        )
        result = time_eval_f(A_data, B_data)
        assert result.mean > 0
        assert len(result.results) == 3


@tvm.testing.requires_micro
def test_autotune():
    """Verify that autotune works with micro."""
    import tvm.relay as relay
    from tvm.micro.testing.utils import check_tune_log

    runtime = Runtime("crt", {"system-lib": True})

    data = relay.var("data", relay.TensorType((1, 3, 64, 64), "float32"))
    weight = relay.var("weight", relay.TensorType((8, 3, 5, 5), "float32"))
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

    main_func = mod["main"]
    shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}

    weight_data = np.ones(shape_dict["weight"]).astype(type_dict["weight"])
    input_data = np.ones(shape_dict["data"]).astype(type_dict["data"])
    params = {"weight": weight_data}
    inputs = {"data": input_data}

    target = tvm.target.target.micro("host")
    template_project_dir = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))

    pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
    with pass_context:
        tasks = tvm.autotvm.task.extract_from_program(mod["main"], {}, target)
    assert len(tasks) > 0

    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=template_project_dir,
        project_options={},
    )
    builder = tvm.autotvm.LocalBuilder(
        n_parallel=1,
        build_kwargs={"build_option": {"tir.disable_vectorize": True}},
        do_fork=True,
        build_func=tvm.micro.autotvm_build_func,
        runtime=runtime,
    )
    runner = tvm.autotvm.LocalRunner(number=1, repeat=1, module_loader=module_loader)

    measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

    tune_log_file = pathlib.Path("crt_autotune.log")
    if tune_log_file.exists():
        tune_log_file.unlink()

    num_trials = 10
    for task in tasks:
        tuner = tvm.autotvm.tuner.GATuner(task)
        tuner.tune(
            n_trial=num_trials,
            measure_option=measure_option,
            callbacks=[
                tvm.autotvm.callback.log_to_file(str(tune_log_file)),
                tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M"),
            ],
            si_prefix="M",
        )
        assert tuner.best_flops > 0

    # TODO(mehrdadh): commented due to autotuning errors
    # check_tune_log(tune_log_file)

    # Build without tuning
    with pass_context:
        lowered = tvm.relay.build(mod, target=TARGET, runtime=runtime, params=params)

    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(template_project_dir, lowered, temp_dir / "project")
    project.build()
    with tvm.micro.Session(project.transport()) as session:
        graph_mod = tvm.micro.create_local_graph_executor(
            lowered.get_graph_json(), session.get_system_lib(), session.device
        )
        graph_mod.set_input(**lowered.get_params())
        graph_mod.run(**inputs)
        expected_output = graph_mod.get_output(0).numpy()
        del graph_mod

    # Build using autotune logs
    with tvm.autotvm.apply_history_best(str(tune_log_file)):
        with pass_context:
            lowered_tuned = tvm.relay.build(mod, target=target, runtime=runtime, params=params)

    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(template_project_dir, lowered_tuned, temp_dir / "project")
    project.build()
    with tvm.micro.Session(project.transport()) as session:
        graph_mod = tvm.micro.create_local_graph_executor(
            lowered_tuned.get_graph_json(), session.get_system_lib(), session.device
        )
        graph_mod.set_input(**lowered_tuned.get_params())
        graph_mod.run(**inputs)
        output = graph_mod.get_output(0).numpy()
        del graph_mod

    tvm.testing.assert_allclose(output, expected_output, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()

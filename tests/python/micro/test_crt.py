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

"""Test C runtime"""

import pathlib
import pytest

import numpy as np

import tvm
import tvm.relay
import tvm.testing
from tvm.target import Target
from tvm.relay.backend import Runtime
from tvm.relay.backend import Executor

pytest.importorskip("pty")

BUILD = True
DEBUG = False

TARGET = tvm.target.target.micro("host")


def _make_sess_from_op(temp_dir, op_name, sched, arg_bufs):
    runtime = Runtime("crt", {"system-lib": True})
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.build(sched, arg_bufs, Target(TARGET, TARGET), runtime=runtime, name=op_name)

    return _make_session(temp_dir, mod)


def _make_session(temp_dir, mod):
    template_project_dir = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
    project = tvm.micro.generate_project(
        template_project_dir, mod, temp_dir / "project", {"verbose": 1}
    )
    project.build()
    project.flash()
    return tvm.micro.Session(project.transport())


def _make_add_sess(temp_dir):
    a = tvm.te.placeholder((2,), dtype="int8")
    b = tvm.te.placeholder((1,), dtype="int8")
    c = tvm.te.compute(a.shape, lambda i: a[i] + b[0], name="c")
    sched = tvm.te.create_schedule(c.op)
    return _make_sess_from_op(temp_dir, "add", sched, [a, b, c])


@tvm.testing.requires_micro
def test_compile_runtime():
    """Test compiling the on-device runtime."""

    temp_dir = tvm.contrib.utils.tempdir()

    with _make_add_sess(temp_dir) as sess:
        a_data = tvm.nd.array(np.array([2, 3], dtype="int8"), device=sess.device)
        assert (a_data.numpy() == np.array([2, 3])).all()
        b_data = tvm.nd.array(np.array([4], dtype="int8"), device=sess.device)
        assert (b_data.numpy() == np.array([4])).all()
        c_data = tvm.nd.array(np.array([0, 0], dtype="int8"), device=sess.device)
        assert (c_data.numpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        system_lib.get_function("add")(a_data, b_data, c_data)
        assert (c_data.numpy() == np.array([6, 7])).all()


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

    temp_dir = tvm.contrib.utils.tempdir()
    relay_mod = tvm.relay.fromtext(
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

        a_data = tvm.nd.array(np.array([2, 3], dtype="uint8"), device=sess.device)
        assert (a_data.numpy() == np.array([2, 3])).all()
        b_data = tvm.nd.array(np.array([4, 7], dtype="uint8"), device=sess.device)
        assert (b_data.numpy() == np.array([4, 7])).all()

        assert graph_mod.get_input_index("a") == 0
        assert graph_mod.get_input_index("b") == 1

        graph_mod.run(a=a_data, b=b_data)

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

    temp_dir = tvm.contrib.utils.tempdir()
    relay_mod = tvm.relay.fromtext(
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
        aot_executor = tvm.micro.create_local_aot_executor(sess)

        assert aot_executor.get_input_index("a") == 0
        assert aot_executor.get_input_index("b") == 1

        assert aot_executor.get_input_name(0) == "a"
        assert aot_executor.get_input_name(1) == "b"

        shape_dict, dtype_dict = aot_executor.get_input_info()
        assert shape_dict == {"a": (1, 2), "b": (1, 2)}
        assert dtype_dict == {"a": "uint8", "b": "uint8"}

        assert aot_executor.get_num_inputs() == 2
        assert aot_executor.get_num_outputs() == 1

        a_np = np.array([[2, 3]], dtype="uint8")
        b_np = np.array([[4, 7]], dtype="uint8")

        aot_executor.get_input("a").copyfrom(a_np)
        b_data = aot_executor.get_input("b").copyfrom(b_np)

        aot_executor.run()

        out = aot_executor.get_output(0)
        assert (out.numpy() == np.array([6, 10])).all()

        b_np_new = np.array([[5, 8]])
        aot_executor.set_input("b", b_np_new)
        assert (b_data.numpy() == b_np_new).all()

    with _make_session(temp_dir, factory) as sess:
        do_test()


@tvm.testing.requires_micro
def test_aot_executor_usmp_const_pool():
    """Test the AOT executor with microTVM using USMP to generate a constant data pool."""

    temp_dir = tvm.contrib.utils.tempdir()
    relay_mod = tvm.relay.fromtext(
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
    c_np = np.array([[8, 9]], dtype="uint8").astype(type_dict["c"])
    params = {"c": c_np}
    with tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True, "tir.usmp.enable": True}
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
            aot_executor = tvm.micro.create_local_aot_executor(sess)
        except tvm._ffi.base.TVMError as excpt:
            raise excpt

        assert aot_executor.get_input_index("a") == 0
        assert aot_executor.get_input_index("b") == 1

        assert aot_executor.get_num_inputs() == 2
        assert aot_executor.get_num_outputs() == 1

        a_np = np.array([[2, 3]], dtype="uint8")
        b_np = np.array([[4, 7]], dtype="uint8")

        aot_executor.get_input("a").copyfrom(a_np)
        b_data = aot_executor.get_input("b").copyfrom(b_np)
        aot_executor.run()

        out = aot_executor.get_output(0)
        assert (out.numpy() == np.array([14, 19])).all()

        b_np_new = np.array([[5, 8]])
        aot_executor.set_input("b", b_np_new)
        assert (b_data.numpy() == b_np_new).all()

    with _make_session(temp_dir, factory) as sess:
        do_test()


@tvm.testing.requires_micro
def test_std_math_functions():
    """Verify that standard math functions can be used."""

    temp_dir = tvm.contrib.utils.tempdir()

    with _make_add_sess(temp_dir) as sess:
        a_data = tvm.nd.array(np.array([2, 3], dtype="int8"), device=sess.device)
        assert (a_data.numpy() == np.array([2, 3])).all()
        b_data = tvm.nd.array(np.array([4], dtype="int8"), device=sess.device)
        assert (b_data.numpy() == np.array([4])).all()
        c_data = tvm.nd.array(np.array([0, 0], dtype="int8"), device=sess.device)
        assert (c_data.numpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        system_lib.get_function("add")(a_data, b_data, c_data)

    temp_dir = tvm.contrib.utils.tempdir()
    a = tvm.te.placeholder((2,), dtype="float32", name="a")
    b = tvm.te.compute(a.shape, lambda i: tvm.te.exp(a[i]), name="b")
    s = tvm.te.create_schedule(b.op)

    with _make_sess_from_op(temp_dir, "myexpf", s, [a, b]) as sess:
        a_data = tvm.nd.array(np.array([2.0, 3.0], dtype="float32"), device=sess.device)
        b_data = tvm.nd.array(np.array([2.0, 3.0], dtype="float32"), device=sess.device)
        lib = sess.get_system_lib()
        func = lib["myexpf"]
        func(a_data, b_data)
        np.testing.assert_allclose(b_data.numpy(), np.array([7.389056, 20.085537]))


@tvm.testing.requires_micro
def test_platform_timer():
    """Verify the platform timer can be used to time remote functions."""

    temp_dir = tvm.contrib.utils.tempdir()
    a = tvm.te.placeholder((2,), dtype="float32", name="a")
    b = tvm.te.compute(a.shape, lambda i: tvm.te.exp(a[i]), name="b")
    s = tvm.te.create_schedule(b.op)

    with _make_sess_from_op(temp_dir, "myexpf", s, [a, b]) as sess:
        a_data = tvm.nd.array(np.array([2.0, 3.0], dtype="float32"), device=sess.device)
        b_data = tvm.nd.array(np.array([2.0, 3.0], dtype="float32"), device=sess.device)
        lib = sess.get_system_lib()
        time_eval_f = lib.time_evaluator(
            "myexpf", sess.device, number=2000, repeat=3, min_repeat_ms=40
        )
        result = time_eval_f(a_data, b_data)
        assert result.mean > 0
        assert len(result.results) == 3


@tvm.testing.requires_micro
def test_autotune():
    """Verify that autotune works with micro."""

    runtime = Runtime("crt", {"system-lib": True})

    data = tvm.relay.var("data", tvm.relay.TensorType((1, 3, 64, 64), "float32"))
    weight = tvm.relay.var("weight", tvm.relay.TensorType((8, 3, 5, 5), "float32"))
    y = tvm.relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = tvm.relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = tvm.relay.transform.InferType()(mod)

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
    with _make_session(temp_dir, lowered) as sess:
        graph_mod = tvm.micro.create_local_graph_executor(
            lowered.get_graph_json(), sess.get_system_lib(), sess.device
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
    with _make_session(temp_dir, lowered_tuned) as sess:
        graph_mod = tvm.micro.create_local_graph_executor(
            lowered_tuned.get_graph_json(), sess.get_system_lib(), sess.device
        )
        graph_mod.set_input(**lowered_tuned.get_params())
        graph_mod.run(**inputs)
        output = graph_mod.get_output(0).numpy()
        del graph_mod

    tvm.testing.assert_allclose(output, expected_output, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()

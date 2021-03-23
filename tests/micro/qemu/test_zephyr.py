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
import os
import subprocess
import sys

import pytest
import numpy as np

import tvm
import tvm.rpc
import tvm.micro
import tvm.relay as relay

from tvm.micro.contrib import zephyr
from tvm.contrib import utils
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end

BUILD = True
DEBUG = False


TARGET = None


def _make_sess_from_op(model, zephyr_board, west_cmd, op_name, sched, arg_bufs):
    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.build(sched, arg_bufs, target, target_host=target, name=op_name)

    return _make_session(model, target, zephyr_board, west_cmd, mod)


def _make_session(model, target, zephyr_board, west_cmd, mod):
    test_name = f"{os.path.splitext(os.path.abspath(__file__))[0]}-{model}"
    prev_build = f"{test_name}-last-build.micro-binary"
    workspace_root = (
        f'{test_name}-workspace/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}'
    )
    workspace_parent = os.path.dirname(workspace_root)
    if not os.path.exists(workspace_parent):
        os.makedirs(workspace_parent)
    workspace = tvm.micro.Workspace(debug=True, root=workspace_root)

    project_dir = os.path.join(os.path.dirname(__file__) or ".", "zephyr-runtime")
    compiler = zephyr.ZephyrCompiler(
        project_dir=project_dir,
        board=zephyr_board,
        zephyr_toolchain_variant="zephyr",
        west_cmd=west_cmd,
    )

    opts = tvm.micro.default_options(f"{project_dir}/crt")
    # TODO(weberlo) verify this is necessary
    opts["bin_opts"]["ccflags"] = ["-std=gnu++14"]
    opts["lib_opts"]["ccflags"] = ["-std=gnu++14"]

    flasher_kw = {}
    if DEBUG:
        flasher_kw["debug_rpc_session"] = tvm.rpc.connect("127.0.0.1", 9090)

    session_kw = {
        "flasher": compiler.flasher(**flasher_kw),
    }

    if BUILD:
        session_kw["binary"] = tvm.micro.build_static_runtime(
            # the x86 compiler *expects* you to give the exact same dictionary for both
            # lib_opts and bin_opts. so the library compiler is mutating lib_opts and
            # the binary compiler is expecting those mutations to be in bin_opts.
            # TODO(weberlo) fix this very bizarre behavior
            workspace,
            compiler,
            mod,
            opts,
        )
        if os.path.exists(prev_build):
            os.unlink(prev_build)
        session_kw["binary"].archive(prev_build, metadata_only=True)
    else:
        unarchive_dir = utils.tempdir()
        session_kw["binary"] = tvm.micro.MicroBinary.unarchive(
            prev_build, unarchive_dir.relpath("binary")
        )

    return tvm.micro.Session(**session_kw)


def _make_add_sess(model, zephyr_board, west_cmd):
    A = tvm.te.placeholder((2,), dtype="int8")
    B = tvm.te.placeholder((1,), dtype="int8")
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)
    return _make_sess_from_op(model, zephyr_board, west_cmd, "add", sched, [A, B, C])


# The models that should pass this configuration. Maps a short, identifying platform string to
# (model, zephyr_board).
PLATFORMS = {
    "host": ("host", "qemu_x86"),
    "stm32f746xx": ("stm32f746xx", "nucleo_f746zg"),
    "nrf5340dk": ("nrf5340dk", "nrf5340dk_nrf5340_cpuapp"),
}


# The same test code can be executed on both the QEMU simulation and on real hardware.
def test_compile_runtime(platform, west_cmd):
    """Test compiling the on-device runtime."""

    model, zephyr_board = PLATFORMS[platform]

    # NOTE: run test in a nested function so cPython will delete arrays before closing the session.
    def test_basic_add(sess):
        A_data = tvm.nd.array(np.array([2, 3], dtype="int8"), device=sess.device)
        assert (A_data.asnumpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype="int8"), device=sess.device)
        assert (B_data.asnumpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype="int8"), device=sess.device)
        assert (C_data.asnumpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        system_lib.get_function("add")(A_data, B_data, C_data)
        assert (C_data.asnumpy() == np.array([6, 7])).all()

    with _make_add_sess(model, zephyr_board, west_cmd) as sess:
        test_basic_add(sess)


def test_platform_timer(platform, west_cmd):
    """Test compiling the on-device runtime."""

    model, zephyr_board = PLATFORMS[platform]

    # NOTE: run test in a nested function so cPython will delete arrays before closing the session.
    def test_basic_add(sess):
        A_data = tvm.nd.array(np.array([2, 3], dtype="int8"), device=sess.device)
        assert (A_data.asnumpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype="int8"), device=sess.device)
        assert (B_data.asnumpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype="int8"), device=sess.device)
        assert (C_data.asnumpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        time_eval_f = system_lib.time_evaluator(
            "add", sess.device, number=20, repeat=3, min_repeat_ms=40
        )
        result = time_eval_f(A_data, B_data, C_data)
        assert (C_data.asnumpy() == np.array([6, 7])).all()
        assert result.mean > 0
        assert len(result.results) == 3

    with _make_add_sess(model, zephyr_board, west_cmd) as sess:
        test_basic_add(sess)


def test_relay(platform, west_cmd):
    """Testing a simple relay graph"""
    model, zephyr_board = PLATFORMS[platform]
    shape = (10,)
    dtype = "int8"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)

    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        graph, mod, params = tvm.relay.build(func, target=target)

    with _make_session(model, target, zephyr_board, west_cmd, mod) as session:
        graph_mod = tvm.micro.create_local_graph_runtime(
            graph, session.get_system_lib(), session.device
        )
        graph_mod.set_input(**params)
        x_in = np.random.randint(10, size=shape[0], dtype=dtype)
        graph_mod.run(x=x_in)
        result = graph_mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(graph_mod.get_input(0).asnumpy(), x_in)
        tvm.testing.assert_allclose(result, x_in * x_in + 1)


class CcompilerAnnotator(ExprMutator):
    """
    This is used to create external functions for ccompiler.
    A simple annotator that creates the following program:
           |
      -- begin --
           |
          add
           |
        subtract
           |
        multiply
           |
       -- end --
           |
    """

    def __init__(self):
        super(CcompilerAnnotator, self).__init__()
        self.in_compiler = 0

    def visit_call(self, call):
        if call.op.name == "add":  # Annotate begin at args
            if self.in_compiler == 1:
                lhs = compiler_begin(super().visit(call.args[0]), "ccompiler")
                rhs = compiler_begin(super().visit(call.args[1]), "ccompiler")
                op = relay.add(lhs, rhs)
                self.in_compiler = 2
                return op
        elif call.op.name == "subtract":
            if self.in_compiler == 1:
                lhs = super().visit(call.args[0])
                rhs = super().visit(call.args[1])
                if isinstance(lhs, relay.expr.Var):
                    lhs = compiler_begin(lhs, "ccompiler")
                if isinstance(rhs, relay.expr.Var):
                    rhs = compiler_begin(rhs, "ccompiler")
                return relay.subtract(lhs, rhs)
        elif call.op.name == "multiply":  # Annotate end at output
            self.in_compiler = 1
            lhs = super().visit(call.args[0])
            rhs = super().visit(call.args[1])
            if isinstance(lhs, relay.expr.Var):
                lhs = compiler_begin(lhs, "ccompiler")
            if isinstance(rhs, relay.expr.Var):
                rhs = compiler_begin(rhs, "ccompiler")
            op = relay.multiply(lhs, rhs)
            if self.in_compiler == 2:
                op = compiler_end(op, "ccompiler")
            self.in_compiler = 0
            return op
        return super().visit_call(call)


def check_result(relay_mod, model, zephyr_board, west_cmd, map_inputs, out_shape, result):
    """Helper function to verify results"""
    TOL = 1e-5
    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        graph, mod, params = tvm.relay.build(relay_mod, target=target)

    with _make_session(model, target, zephyr_board, west_cmd, mod) as session:
        rt_mod = tvm.micro.create_local_graph_runtime(
            graph, session.get_system_lib(), session.device
        )
        rt_mod.set_input(**params)
        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**params)
        rt_mod.run()

        out_shapes = out_shape if isinstance(out_shape, list) else [out_shape]
        results = result if isinstance(result, list) else [result]

        for idx, shape in enumerate(out_shapes):
            out = tvm.nd.empty(shape, device=session.device)
            out = rt_mod.get_output(idx, out)
            tvm.testing.assert_allclose(out.asnumpy(), results[idx], rtol=TOL, atol=TOL)


def test_byoc_utvm(platform, west_cmd):
    """This is a simple test case to check BYOC capabilities of uTVM"""
    model, zephyr_board = PLATFORMS[platform]
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
    ann = CcompilerAnnotator()
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
        zephyr_board=zephyr_board,
        west_cmd=west_cmd,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.dirname(__file__)] + sys.argv[1:]))

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
import glob
import os
import pytest

pytest.importorskip("pty")
import sys
import subprocess
import textwrap

import numpy as np
import pytest

import tvm
import tvm.relay
import tvm.testing
from tvm.target import Target

from tvm.topi.utils import get_const_tuple
from tvm.topi.testing import conv2d_nchw_python

BUILD = True
DEBUG = False

TARGET = tvm.target.target.micro("host")


def _make_sess_from_op(workspace, op_name, sched, arg_bufs):
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.build(sched, arg_bufs, Target(TARGET, TARGET), name=op_name)

    return _make_session(workspace, mod)


def _make_session(workspace, mod):
    template_project_dir = os.path.join(tvm.micro.get_standalone_crt_dir(), "template", "host")
    project = tvm.micro.generate_project(template_project_dir, mod, workspace.relpath("project"), {"verbose": 1})
    project.build()
    project.flash()
    return tvm.micro.Session(project.transport())


def _make_add_sess(workspace):
    A = tvm.te.placeholder((2,), dtype="int8")
    B = tvm.te.placeholder((1,), dtype="int8")
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)
    return _make_sess_from_op(workspace, "add", sched, [A, B, C])


def _make_ident_sess(workspace):
    A = tvm.te.placeholder((2,), dtype="int8")
    B = tvm.te.compute(A.shape, lambda i: A[i], name="B")
    sched = tvm.te.create_schedule(B.op)
    return _make_sess_from_op(workspace, "ident", sched, [A, B])


@tvm.testing.requires_micro
def test_compile_runtime():
    """Test compiling the on-device runtime."""
    import tvm.micro

    workspace = tvm.micro.Workspace()

    with _make_add_sess(workspace) as sess:
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

    workspace = tvm.micro.Workspace()

    with _make_add_sess(workspace) as sess:
        try:
            sess._rpc.get_function("tvm.testing.reset_server")()
            assert False, "expected to raise SessionTerminatedError; did not raise"
        except tvm.micro.SessionTerminatedError:
            pass


@tvm.testing.requires_micro
def test_graph_executor():
    """Test use of the graph executor with microTVM."""
    import tvm.micro

    workspace = tvm.micro.Workspace(debug=True)
    relay_mod = tvm.parser.fromtext(
        """
      #[version = "0.0.5"]
      def @main(%a : Tensor[(1, 2), uint8], %b : Tensor[(1, 2), uint8]) {
          %0 = %a + %b;
          %0
      }"""
    )

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory = tvm.relay.build(relay_mod, target=TARGET)

    with _make_session(workspace, factory) as sess:
        graph_mod = tvm.micro.create_local_graph_executor(
            factory.get_graph_json(), sess.get_system_lib(), sess.device
        )
        A_data = tvm.nd.array(np.array([2, 3], dtype="uint8"), device=sess.device)
        assert (A_data.numpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4, 7], dtype="uint8"), device=sess.device)
        assert (B_data.numpy() == np.array([4, 7])).all()

        graph_mod.run(a=A_data, b=B_data)

        out = graph_mod.get_output(0)
        assert (out.numpy() == np.array([6, 10])).all()


@tvm.testing.requires_micro
def test_std_math_functions():
    """Verify that standard math functions can be used."""
    import tvm.micro

    workspace = tvm.micro.Workspace()

    with _make_add_sess(workspace) as sess:
        A_data = tvm.nd.array(np.array([2, 3], dtype="int8"), device=sess.device)
        assert (A_data.numpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype="int8"), device=sess.device)
        assert (B_data.numpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype="int8"), device=sess.device)
        assert (C_data.numpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        system_lib.get_function("add")(A_data, B_data, C_data)

    workspace = tvm.micro.Workspace()
    A = tvm.te.placeholder((2,), dtype="float32", name="A")
    B = tvm.te.compute(A.shape, lambda i: tvm.te.exp(A[i]), name="B")
    s = tvm.te.create_schedule(B.op)

    with _make_sess_from_op(workspace, "myexpf", s, [A, B]) as sess:
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

    workspace = tvm.micro.Workspace()
    A = tvm.te.placeholder((2,), dtype="float32", name="A")
    B = tvm.te.compute(A.shape, lambda i: tvm.te.exp(A[i]), name="B")
    s = tvm.te.create_schedule(B.op)

    with _make_sess_from_op(workspace, "myexpf", s, [A, B]) as sess:
        A_data = tvm.nd.array(np.array([2.0, 3.0], dtype="float32"), device=sess.device)
        B_data = tvm.nd.array(np.array([2.0, 3.0], dtype="float32"), device=sess.device)
        lib = sess.get_system_lib()
        time_eval_f = lib.time_evaluator(
            "myexpf", sess.device, number=2000, repeat=3, min_repeat_ms=40
        )
        result = time_eval_f(A_data, B_data)
        assert result.mean > 0
        assert len(result.results) == 3


if __name__ == "__main__":
    test_graph_executor()
#     sys.exit(pytest.main([__file__] + sys.argv[1:]))

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
import pty
import sys
import subprocess
import textwrap

import numpy as np

import tvm
import tvm.relay
import tvm.micro
from tvm.micro import transport

from tvm.topi.util import get_const_tuple
from tvm.topi.testing import conv2d_nchw_python

BUILD = True
DEBUG = False

TARGET = tvm.target.target.micro("host")


def _make_sess_from_op(workspace, op_name, sched, arg_bufs):
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.build(sched, arg_bufs, TARGET, target_host=TARGET, name=op_name)

    return _make_session(workspace, mod)


def _make_session(workspace, mod):
    compiler = tvm.micro.DefaultCompiler(target=TARGET)
    opts = tvm.micro.default_options(os.path.join(tvm.micro.CRT_ROOT_DIR, "host"))

    micro_binary = tvm.micro.build_static_runtime(
        # the x86 compiler *expects* you to give the exact same dictionary for both
        # lib_opts and bin_opts. so the library compiler is mutating lib_opts and
        # the binary compiler is expecting those mutations to be in bin_opts.
        # TODO(weberlo) fix this very bizarre behavior
        workspace,
        compiler,
        mod,
        lib_opts=opts["bin_opts"],
        bin_opts=opts["bin_opts"],
    )

    flasher_kw = {
        "debug": DEBUG,
    }
    flasher = compiler.flasher(**flasher_kw)
    return tvm.micro.Session(binary=micro_binary, flasher=flasher)


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


def test_compile_runtime():
    """Test compiling the on-device runtime."""
    workspace = tvm.micro.Workspace()

    with _make_add_sess(workspace) as sess:
        A_data = tvm.nd.array(np.array([2, 3], dtype="int8"), ctx=sess.context)
        assert (A_data.asnumpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype="int8"), ctx=sess.context)
        assert (B_data.asnumpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype="int8"), ctx=sess.context)
        assert (C_data.asnumpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        system_lib.get_function("add")(A_data, B_data, C_data)
        assert (C_data.asnumpy() == np.array([6, 7])).all()


def test_reset():
    """Test when the remote end resets during a session."""
    workspace = tvm.micro.Workspace()

    with _make_add_sess(workspace) as sess:
        try:
            sess._rpc.get_function("tvm.testing.reset_server")()
            assert False, "expected to raise SessionTerminatedError; did not raise"
        except transport.SessionTerminatedError:
            pass


def test_graph_runtime():
    """Test use of the graph runtime with microTVM."""
    workspace = tvm.micro.Workspace()
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

    with _make_session(workspace, factory.get_lib()) as sess:
        graph_mod = tvm.micro.create_local_graph_runtime(
            factory.get_json(), sess.get_system_lib(), sess.context
        )
        A_data = tvm.nd.array(np.array([2, 3], dtype="uint8"), ctx=sess.context)
        assert (A_data.asnumpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4, 7], dtype="uint8"), ctx=sess.context)
        assert (B_data.asnumpy() == np.array([4, 7])).all()

        graph_mod.run(a=A_data, b=B_data)

        out = graph_mod.get_output(0)
        assert (out.asnumpy() == np.array([6, 10])).all()


if __name__ == "__main__":
    test_compile_runtime()
    test_reset()
    test_graph_runtime()

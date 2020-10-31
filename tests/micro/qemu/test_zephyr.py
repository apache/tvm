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
import tvm.relay

from tvm.micro.contrib import zephyr
from tvm.contrib import utils

BUILD = True
DEBUG = False


TARGET = None


def _make_sess_from_op(model, zephyr_board, op_name, sched, arg_bufs):
    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.build(sched, arg_bufs, target, target_host=target, name=op_name)

    return _make_session(model, target, zephyr_board, mod)


def _make_session(model, target, zephyr_board, mod):
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
        board="nucleo_f746zg" if "stm32f746" in str(target) else "qemu_x86",
        zephyr_toolchain_variant="zephyr",
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
            lib_opts=opts["lib_opts"],
            bin_opts=opts["bin_opts"],
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


def _make_add_sess(model, zephyr_board):
    A = tvm.te.placeholder((2,), dtype="int8")
    B = tvm.te.placeholder((1,), dtype="int8")
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)
    return _make_sess_from_op(model, zephyr_board, "add", sched, [A, B, C])


# The models that should pass this configuration. Maps a short, identifying platform string to
# (model, zephyr_board).
PLATFORMS = {
    "host": ("host", "qemu_x86"),
    "stm32f746xx": ("stm32f746xx", "nucleo_f746zg"),
}


# The same test code can be executed on both the QEMU simulation and on real hardware.
def test_compile_runtime(platform):
    """Test compiling the on-device runtime."""

    model, zephyr_board = PLATFORMS[platform]

    # NOTE: run test in a nested function so cPython will delete arrays before closing the session.
    def test_basic_add(sess):
        A_data = tvm.nd.array(np.array([2, 3], dtype="int8"), ctx=sess.context)
        assert (A_data.asnumpy() == np.array([2, 3])).all()
        B_data = tvm.nd.array(np.array([4], dtype="int8"), ctx=sess.context)
        assert (B_data.asnumpy() == np.array([4])).all()
        C_data = tvm.nd.array(np.array([0, 0], dtype="int8"), ctx=sess.context)
        assert (C_data.asnumpy() == np.array([0, 0])).all()

        system_lib = sess.get_system_lib()
        system_lib.get_function("add")(A_data, B_data, C_data)
        assert (C_data.asnumpy() == np.array([6, 7])).all()

    with _make_add_sess(model, zephyr_board) as sess:
        test_basic_add(sess)


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.dirname(__file__)] + sys.argv[1:]))

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
from hashlib import new
import logging
import os
import subprocess
import sys
import logging
import pathlib

import pytest
import numpy as np
from PIL import Image

import tvm
import tvm.rpc
import tvm.micro
import tvm.relay as relay

from tvm.micro.contrib import zephyr
from tvm.contrib import utils
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end

import conftest

_LOG = logging.getLogger(__name__)

PLATFORMS = conftest.PLATFORMS

# If set, build the uTVM binary from scratch on each test.
# Otherwise, reuses the build from the previous test run.
BUILD = True

# If set, enable a debug session while the test is running.
# Before running the test, in a separate shell, you should run:
#   python -m tvm.exec.microtvm_debug_shell
DEBUG = False


def _build_session_kw(model, target, zephyr_board, west_cmd, mod, runtime_path):
    parent_dir = os.path.dirname(__file__)
    filename = os.path.splitext(os.path.basename(__file__))[0]
    prev_build = f"{os.path.join(parent_dir, 'archive')}_{filename}_{zephyr_board}_last_build.micro"
    workspace_root = os.path.join(
        f"{os.path.join(parent_dir, 'workspace')}_{filename}_{zephyr_board}",
        datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
    )
    workspace_parent = os.path.dirname(workspace_root)
    if not os.path.exists(workspace_parent):
        os.makedirs(workspace_parent)
    workspace = tvm.micro.Workspace(debug=True, root=workspace_root)

    compiler = zephyr.ZephyrCompiler(
        project_dir=runtime_path,
        board=zephyr_board,
        zephyr_toolchain_variant="zephyr",
        west_cmd=west_cmd,
        env_vars={"ZEPHYR_RUNTIME": "ZEPHYR-AOT"},
    )

    opts = tvm.micro.default_options(os.path.join(runtime_path, "crt"))
    opts["bin_opts"]["include_dirs"].append(os.path.join(runtime_path, "include"))
    opts["lib_opts"]["include_dirs"].append(os.path.join(runtime_path, "include"))

    flasher_kw = {}
    if DEBUG:
        flasher_kw["debug_rpc_session"] = tvm.rpc.connect("127.0.0.1", 9090)

    session_kw = {
        "flasher": compiler.flasher(**flasher_kw),
    }

    if BUILD:
        session_kw["binary"] = tvm.micro.build_static_runtime(
            workspace,
            compiler,
            mod,
            opts,
            runtime="zephyr-aot",
            extra_libs=[tvm.micro.get_standalone_crt_lib("memory")],
        )
        if os.path.exists(prev_build):
            os.unlink(prev_build)
        session_kw["binary"].archive(prev_build, metadata_only=True)
    else:
        unarchive_dir = utils.tempdir()
        session_kw["binary"] = tvm.micro.MicroBinary.unarchive(
            prev_build, unarchive_dir.relpath("binary")
        )

    return session_kw


def _create_header_file(tensor_name, npy_data, output_path):
    """
    This method generates a header file containing the data contained in the numpy array provided.
    It is used to capture the tensor data (for both inputs and expected outputs).
    """
    file_path = pathlib.Path(f"{output_path}/" + tensor_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write("#include <stddef.h>\n")
        header_file.write("#include <stdint.h>\n")
        header_file.write("#include <dlpack/dlpack.h>\n")
        header_file.write(f"const size_t {tensor_name}_len = {npy_data.size};\n")

        if npy_data.dtype == "int8":
            header_file.write(f"int8_t {tensor_name}[] =")
        elif npy_data.dtype == "int32":
            header_file.write(f"int32_t {tensor_name}[] = ")
        elif npy_data.dtype == "uint8":
            header_file.write(f"uint8_t {tensor_name}[] = ")
        elif npy_data.dtype == "float32":
            header_file.write(f"float {tensor_name}[] = ")

        header_file.write("{")
        for i in np.ndindex(npy_data.shape):
            header_file.write(f"{npy_data[i]}, ")
        header_file.write("};\n\n")


def _read_line(fd):
    data = ""
    new_c = ""
    while new_c != "\n":
        new_c = fd.read(1, timeout_sec=10)
        new_c = new_c.decode("ascii")
        data = data + new_c
    return data


def _get_result_line(fd):
    while True:
        data = _read_line(fd)
        if "result" in data:
            return data


def test_tflite(platform, west_cmd):
    """Testing a TFLite model."""
    model, zephyr_board = PLATFORMS[platform]
    input_shape = (1, 32, 32, 3)
    output_shape = (1, 10)

    this_dir = os.path.dirname(__file__)
    tvm_source_dir = os.path.join(this_dir, "..", "..", "..")
    runtime_path = os.path.join(tvm_source_dir, "apps", "microtvm", "zephyr", "aot_demo")
    model_path = os.path.join(this_dir, "testdata", "ic_fp32.tflite")

    # Import TFLite model
    tflite_model_buf = open(model_path, "rb").read()
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # Load TFLite model and convert to Relay
    relay_mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict={"input_1": input_shape}, dtype_dict={"input_1 ": "float32"}
    )

    target = tvm.target.target.micro(model, options=["-link-params=1", "--executor=aot"])
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params)

    # Load sample and generate input/expected_output header files
    sample = np.load(os.path.join(this_dir, "testdata", "ic_sample_fp32_8.npy"))
    model_files_path = os.path.join(runtime_path, "include")
    _create_header_file((f"input_data"), sample, model_files_path)
    _create_header_file(
        "output_data", np.zeros(shape=output_shape, dtype="float32"), model_files_path
    )

    session_kw = _build_session_kw(model, target, zephyr_board, west_cmd, lowered.lib, runtime_path)
    transport = session_kw["flasher"].flash(session_kw["binary"])
    transport.open()

    result_line = _get_result_line(transport)
    result_line.strip("\n")
    result = int(result_line.split(":")[1])
    assert result == 8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))

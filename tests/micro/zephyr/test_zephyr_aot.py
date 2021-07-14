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

import datetime
from hashlib import new
import logging
import os
import sys
import logging
import pathlib

import pytest
import numpy as np

import tvm
import tvm.rpc
import tvm.micro
import tvm.testing
import tvm.relay as relay

from tvm.micro.contrib import zephyr
from tvm.contrib import utils
from tvm.contrib.download import download_testdata

import conftest

_LOG = logging.getLogger(__name__)

PLATFORMS = conftest.PLATFORMS


def _build_session_kw(model, target, zephyr_board, west_cmd, mod, runtime_path, build_config):
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
    if build_config["debug"]:
        flasher_kw["debug_rpc_session"] = tvm.rpc.connect("127.0.0.1", 9090)

    session_kw = {
        "flasher": compiler.flasher(**flasher_kw),
    }

    if not build_config["skip_build"]:
        session_kw["binary"] = tvm.micro.build_static_runtime(
            workspace,
            compiler,
            mod,
            opts,
            executor="aot",
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
        else:
            raise ValueError("Data type not expected.")

        header_file.write("{")
        for i in np.ndindex(npy_data.shape):
            header_file.write(f"{npy_data[i]}, ")
        header_file.write("};\n\n")


def _read_line(fd):
    data = ""
    new_line = False
    while True:
        if new_line:
            break
        new_data = fd.read(1, timeout_sec=10)
        logging.debug(f"read data: {new_data}")
        for item in new_data:
            new_c = chr(item)
            data = data + new_c
            if new_c == "\n":
                new_line = True
                break
    return data


def _get_message(fd, expr: str):
    while True:
        data = _read_line(fd)
        logging.debug(f"new line: {data}")
        if expr in data:
            return data


@tvm.testing.requires_micro
def test_tflite(platform, west_cmd, skip_build, tvm_debug):
    """Testing a TFLite model."""
    model, zephyr_board = PLATFORMS[platform]
    input_shape = (1, 32, 32, 3)
    output_shape = (1, 10)
    build_config = {"skip_build": skip_build, "debug": tvm_debug}

    this_dir = os.path.dirname(__file__)
    tvm_source_dir = os.path.join(this_dir, "..", "..", "..")
    runtime_path = os.path.join(tvm_source_dir, "apps", "microtvm", "zephyr", "aot_demo")
    model_url = "https://github.com/eembc/ulpmark-ml/raw/fc1499c7cc83681a02820d5ddf5d97fe75d4f663/base_models/ic01/ic01_fp32.tflite"
    model_path = download_testdata(model_url, "ic01_fp32.tflite", module="model")

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

    target = tvm.target.target.micro(
        model, options=["-link-params=1", "--executor=aot", "--unpacked-api=1"]
    )
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params)

    # Load sample and generate input/output header files
    sample_url = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/testdata_image_classification_fp32_8.npy"
    sample_path = download_testdata(
        sample_url, "testdata_image_classification_fp32_8.npy", module="data"
    )
    sample = np.load(sample_path)
    model_files_path = os.path.join(runtime_path, "include")
    _create_header_file((f"input_data"), sample, model_files_path)
    _create_header_file(
        "output_data", np.zeros(shape=output_shape, dtype="float32"), model_files_path
    )

    session_kw = _build_session_kw(
        model, target, zephyr_board, west_cmd, lowered.lib, runtime_path, build_config
    )
    transport = session_kw["flasher"].flash(session_kw["binary"])
    transport.open()
    transport.write(b"start\n", timeout_sec=5)

    result_line = _get_message(transport, "#result")
    result_line = result_line.strip("\n")
    result_line = result_line.split(":")
    result = int(result_line[1])
    time = int(result_line[2])
    logging.info(f"Result: {result}\ttime: {time} ms")
    assert result == 8


@tvm.testing.requires_micro
def test_qemu_make_fail(platform, west_cmd, skip_build, tvm_debug):
    if platform not in ["host", "mps2_an521"]:
        pytest.skip(msg="Only for QEMU targets.")

    """Testing QEMU make fail."""
    model, zephyr_board = PLATFORMS[platform]
    build_config = {"skip_build": skip_build, "debug": tvm_debug}
    shape = (10,)
    dtype = "float32"

    this_dir = pathlib.Path(__file__).parent
    tvm_source_dir = this_dir / ".." / ".." / ".."
    runtime_path = tvm_source_dir / "apps" / "microtvm" / "zephyr" / "aot_demo"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)

    target = tvm.target.target.micro(model, options=["-link-params=1", "--executor=aot"])
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(func, target)

    # Generate input/output header files
    model_files_path = os.path.join(runtime_path, "include")
    _create_header_file((f"input_data"), np.zeros(shape=shape, dtype=dtype), model_files_path)
    _create_header_file("output_data", np.zeros(shape=shape, dtype=dtype), model_files_path)

    session_kw = _build_session_kw(
        model, target, zephyr_board, west_cmd, lowered.lib, runtime_path, build_config
    )

    file_path = os.path.join(session_kw["binary"].base_dir, "zephyr/CMakeFiles/run.dir/build.make")
    assert os.path.isfile(file_path), f"[{file_path}] does not exist."

    # Remove a file to create make failure.
    os.remove(file_path)
    transport = session_kw["flasher"].flash(session_kw["binary"])
    with pytest.raises(RuntimeError) as excinfo:
        transport.open()
    assert "QEMU setup failed" in str(excinfo.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))

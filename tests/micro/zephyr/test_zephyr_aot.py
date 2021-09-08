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

import io
import logging
import os
import sys
import logging
import pathlib
import tarfile
import tempfile

import pytest
import numpy as np

import tvm
import tvm.rpc
import tvm.micro
from tvm.micro.project_api import server
import tvm.testing
import tvm.relay as relay

from tvm.contrib import utils
from tvm.contrib.download import download_testdata
from tvm.micro.interface_api import generate_c_interface_header

import conftest

_LOG = logging.getLogger(__name__)


def _build_project(temp_dir, zephyr_board, west_cmd, mod, build_config, extra_files_tar=None):
    template_project_dir = (
        pathlib.Path(__file__).parent
        / ".."
        / ".."
        / ".."
        / "apps"
        / "microtvm"
        / "zephyr"
        / "template_project"
    ).resolve()
    project_dir = temp_dir / "project"
    project = tvm.micro.generate_project(
        str(template_project_dir),
        mod,
        project_dir,
        {
            "extra_files_tar": extra_files_tar,
            "project_type": "aot_demo",
            "west_cmd": west_cmd,
            "verbose": bool(build_config.get("debug")),
            "zephyr_board": zephyr_board,
        },
    )
    project.build()
    return project, project_dir


def _create_header_file(tensor_name, npy_data, output_path, tar_file):
    """
    This method generates a header file containing the data contained in the numpy array provided.
    It is used to capture the tensor data (for both inputs and expected outputs).
    """
    header_file = io.StringIO()
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

    header_file_bytes = bytes(header_file.getvalue(), "utf-8")
    raw_path = pathlib.Path(output_path) / f"{tensor_name}.h"
    ti = tarfile.TarInfo(name=str(raw_path))
    ti.size = len(header_file_bytes)
    ti.mode = 0o644
    ti.type = tarfile.REGTYPE
    tar_file.addfile(ti, io.BytesIO(header_file_bytes))


def _read_line(fd, timeout_sec: int):
    data = ""
    new_line = False
    while True:
        if new_line:
            break
        new_data = fd.read(1, timeout_sec=timeout_sec)
        logging.debug(f"read data: {new_data}")
        for item in new_data:
            new_c = chr(item)
            data = data + new_c
            if new_c == "\n":
                new_line = True
                break
    return data


def _get_message(fd, expr: str, timeout_sec: int):
    while True:
        data = _read_line(fd, timeout_sec)
        logging.debug(f"new line: {data}")
        if expr in data:
            return data


@tvm.testing.requires_micro
def test_tflite(temp_dir, board, west_cmd, tvm_debug):
    """Testing a TFLite model."""

    if board not in ["qemu_x86", "mps2_an521", "nrf5340dk", "stm32l4r5zi_nucleo", "zynq_mp_r5"]:
        pytest.skip(msg="Model does not fit.")

    model = conftest.ZEPHYR_BOARDS[board]
    input_shape = (1, 32, 32, 3)
    output_shape = (1, 10)
    build_config = {"debug": tvm_debug}

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
        model, options=["-link-params=1", "--executor=aot", "--unpacked-api=1", "--interface-api=c"]
    )
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params)

    # Load sample and generate input/output header files
    sample_url = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/testdata_image_classification_fp32_8.npy"
    sample_path = download_testdata(
        sample_url, "testdata_image_classification_fp32_8.npy", module="data"
    )
    sample = np.load(sample_path)

    with tempfile.NamedTemporaryFile() as tar_temp_file:
        with tarfile.open(tar_temp_file.name, "w:gz") as tf:
            with tempfile.TemporaryDirectory() as tar_temp_dir:
                model_files_path = os.path.join(tar_temp_dir, "include")
                os.mkdir(model_files_path)
                header_path = generate_c_interface_header(
                    lowered.libmod_name, ["input_1"], ["output"], model_files_path
                )
                tf.add(header_path, arcname=os.path.relpath(header_path, tar_temp_dir))

            _create_header_file("input_data", sample, "include", tf)
            _create_header_file(
                "output_data", np.zeros(shape=output_shape, dtype="float32"), "include", tf
            )

        project, _ = _build_project(
            temp_dir,
            board,
            west_cmd,
            lowered,
            build_config,
            extra_files_tar=tar_temp_file.name,
        )

    project.flash()
    with project.transport() as transport:
        timeout_read = 60
        _get_message(transport, "#wakeup", timeout_sec=timeout_read)
        transport.write(b"start\n", timeout_sec=5)
        result_line = _get_message(transport, "#result", timeout_sec=timeout_read)

    result_line = result_line.strip("\n")
    result_line = result_line.split(":")
    result = int(result_line[1])
    time = int(result_line[2])
    logging.info(f"Result: {result}\ttime: {time} ms")
    assert result == 8


@tvm.testing.requires_micro
def test_qemu_make_fail(temp_dir, board, west_cmd, tvm_debug):
    """Testing QEMU make fail."""
    if board not in ["qemu_x86", "mps2_an521"]:
        pytest.skip(msg="Only for QEMU targets.")

    model = conftest.ZEPHYR_BOARDS[board]
    build_config = {"debug": tvm_debug}
    shape = (10,)
    dtype = "float32"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)
    ir_mod = tvm.IRModule.from_expr(func)

    target = tvm.target.target.micro(model, options=["-link-params=1", "--executor=aot"])
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(ir_mod, target)

    # Generate input/output header files
    with tempfile.NamedTemporaryFile() as tar_temp_file:
        with tarfile.open(tar_temp_file.name, "w:gz") as tf:
            with tempfile.TemporaryDirectory() as tar_temp_dir:
                model_files_path = os.path.join(tar_temp_dir, "include")
                os.mkdir(model_files_path)
                header_path = generate_c_interface_header(
                    lowered.libmod_name, ["input_1"], ["output"], model_files_path
                )
                tf.add(header_path, arcname=os.path.relpath(header_path, tar_temp_dir))
            _create_header_file("input_data", np.zeros(shape=shape, dtype=dtype), "include", tf)
            _create_header_file("output_data", np.zeros(shape=shape, dtype=dtype), "include", tf)

        project, project_dir = _build_project(
            temp_dir,
            board,
            west_cmd,
            lowered,
            build_config,
            extra_files_tar=tar_temp_file.name,
        )

    file_path = (
        pathlib.Path(project_dir) / "build" / "zephyr" / "CMakeFiles" / "run.dir" / "build.make"
    )
    assert file_path.is_file(), f"[{file_path}] does not exist."

    # Remove a file to create make failure.
    os.remove(file_path)
    project.flash()
    with pytest.raises(server.JSONRPCError) as excinfo:
        project.transport().open()
    assert "QEMU setup failed" in str(excinfo.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))

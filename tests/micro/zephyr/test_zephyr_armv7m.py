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
import pathlib
import sys
import logging
import tarfile
import tempfile

import pytest
import numpy as np

import tvm
import tvm.rpc
import tvm.micro
import tvm.testing
import tvm.relay as relay

from tvm.contrib.download import download_testdata
from tvm.micro.interface_api import generate_c_interface_header

import conftest

_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PLATFORMS = conftest.PLATFORMS

TEMPLATE_PROJECT_DIR = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / ".."
    / "apps"
    / "microtvm"
    / "zephyr"
    / "template_project"
).resolve()


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




def _open_tflite_model(model_path: str):
    # Import TFLite model
    tflite_model_buf = open(model_path, "rb").read()
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    relay_mod, params = relay.frontend.from_tflite(tflite_model)
    #relay_mod, params = relay.frontend.from_tflite(
    #    tflite_model, shape_dict={"input_1": input_shape}, dtype_dict={"input_1 ": "int8"}
    #)

    return relay_mod, params

def _get_test_data(testdata_dir):

    from PIL import Image
    image_files = ["digit-2.jpg"]

    for file in image_files:
        img = Image.open(testdata_dir / file).resize((28, 28))
        # img = np.asarray(img).astype("float32")
        img = np.asarray(img).astype("uint8")
        sample = np.reshape(img, -1)

    output_shape = (1, 10)

    return sample, output_shape


def _apply_desired_layout_isa(relay_mod):

    desired_layouts = {'qnn.conv2d': ['NHWC', 'HWOI'], 'nn.conv2d': ['NHWC', 'HWOI']}

    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)])

    with tvm.transform.PassContext(opt_level=3):
        return seq(relay_mod)

def _apply_desired_layout_no_isa(relay_mod):

    desired_layouts = {'qnn.conv2d': ['NHWC', 'HWIO'], 'nn.conv2d': ['NHWC', 'HWIO']}

    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)])

    with tvm.transform.PassContext(opt_level=3):
        return seq(relay_mod)

def _generate_project(temp_dir, zephyr_board, west_cmd, lowered, build_config, sample, output_shape):

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
            _create_header_file("output_data", np.zeros(shape=output_shape, dtype="float32"), "include", tf)

        project, _ = _build_project(
            temp_dir,
            zephyr_board,
            west_cmd,
            lowered,
            build_config,
            extra_files_tar=tar_temp_file.name,
        )

    return project


def _run_model(temp_dir, zephyr_board, west_cmd, lowered, build_config, sample, output_shape):

    project = _generate_project(temp_dir, zephyr_board, west_cmd, lowered, build_config, sample, output_shape)

    project.flash()

    with project.transport() as transport:
        timeout_read = 60
        # _get_message(transport, "#wakeup", timeout_sec=timeout_read)
        # logging.debug("Wakeup received")
        transport.write(b"start\n", timeout_sec=5)
        result_line = _get_message(transport, "#result", timeout_sec=timeout_read)

    result_line = result_line.strip("\n")
    result_line = result_line.split(":")
    result = int(result_line[1])
    time = int(result_line[2])
    logging.info(f"Result: {result}\ttime: {time} ms")

    return result, time


@tvm.testing.requires_micro
def test_armv7m_intrinsic(temp_dir, platform, west_cmd, tvm_debug):
    from tvm.relay.op import op as reg
    from tvm.relay.qnn.op import layout_conversions

    @reg.register_convert_op_layout("qnn.conv2d", level=11)
    def convert_conv2d(attrs, inputs, tinfos, desired_layouts):
        channels = tinfos[0].shape[attrs["data_layout"].find("C")]
        if channels % 4 != 0:
            return None
        return layout_conversions.convert_qnn_conv2d(attrs, inputs, tinfos, desired_layouts)

    """Testing a ARM v7m ISA extension."""

    if platform not in ["nrf5340dk", "stm32f746xx_disco", "stm32f746xx_nucleo", "stm32l4r5zi_nucleo"]:
        pytest.skip(msg="Platform does not support ARM v7m ISA extenion.")

    model, zephyr_board = PLATFORMS[platform]

    build_config = {"debug": tvm_debug}

    this_dir = pathlib.Path(os.path.dirname(__file__))
    testdata_dir = this_dir.parent / "testdata" / "armv7m"

    relay_mod, params = _open_tflite_model(testdata_dir / "mnist_model_quant_conv3.tflite")

    sample, output_shape = _get_test_data(testdata_dir)

    relay_mod_isa = _apply_desired_layout_isa(relay_mod)
    # kernel layout "HWIO" not supported by isa extension for now
    relay_mod_no_isa = _apply_desired_layout_no_isa(relay_mod)

    target_isa = tvm.target.target.micro(
        model, options=["-keys=arm_cpu,cpu", "-march=armv7e-m", "-link-params=1", "--executor=aot", "--unpacked-api=1", "--interface-api=c"]
    )

    target_no_isa = tvm.target.target.micro(
        model, options=["-keys=arm_cpu,cpu", "-link-params=1", "--executor=aot", "--unpacked-api=1", "--interface-api=c"]
    )

    temp_dir_isa = temp_dir / "isa"
    temp_dir_no_isa = temp_dir / "noisa"

    os.makedirs(temp_dir_isa, exist_ok=True)
    os.makedirs(temp_dir_no_isa, exist_ok=True)

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered_isa = relay.build(relay_mod_isa, target_isa, params=params)
        lowered_no_isa = relay.build(relay_mod_no_isa, target_no_isa, params=params)
        result_isa, time_isa = _run_model(temp_dir_isa, zephyr_board, west_cmd, lowered_isa, build_config, sample, output_shape)
        result_no_isa, time_no_isa = _run_model(temp_dir_no_isa, zephyr_board, west_cmd, lowered_no_isa, build_config, sample, output_shape)

    assert result_no_isa == result_isa
    assert time_no_isa > time_isa


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))

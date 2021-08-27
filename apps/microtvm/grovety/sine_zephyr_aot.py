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

import logging
import os
import logging
import pathlib
import numpy as np

import tvm
import tvm.rpc
import tvm.micro
import tvm.testing
import tvm.relay as relay

from tvm.micro.interface_api import generate_c_interface_header

from common import *

_LOG = logging.getLogger(__name__)

logging.basicConfig(level="DEBUG")

TEMPLATE_PROJECT_DIR = tvm_repo_root() + "/apps/microtvm/grovety/template_project"

verbose = True
platform = "stm32f746xx_nucleo"


def _create_header_file(tensor_name, npy_data, output_path):
    """
    This method generates a header file containing the data contained in the numpy array provided.
    It is used to capture the tensor data (for both inputs and expected outputs).
    """
    file_path = pathlib.Path(f"{output_path}/" + tensor_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".h").resolve()
    print(f"raw path = {raw_path}")
    with open(raw_path, "w") as header_file:
        header_file.write("#include <stddef.h>\n")
        header_file.write("#include <stdint.h>\n")
        header_file.write("#include <dlpack/dlpack.h>\n")
        header_file.write("#define input_1 dense_4_input\n")
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



if __name__ == "__main__":
    workspace_dir = create_workspace_dir(platform, 'sine_zephyr_aot', mkdir=False)
    model, zephyr_board = PLATFORMS[platform]

    sine_model_path = download_sine_model()

    relay_mod, params, input = open_tflite_model(sine_model_path)
    input_tensor, input_shape, input_dtype = input



    target = tvm.target.target.micro(model, options=["-link-params=1", "--executor=aot", "--unpacked-api=1", "--interface-api=c"])
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params)

    project = tvm.micro.generate_project(
        str(TEMPLATE_PROJECT_DIR),
        lowered,
        workspace_dir,
        {
            "project_type": "aot_demo",
            "west_cmd": "west",
            "verbose": verbose,
            "zephyr_board": zephyr_board,
        },
    )

    os.makedirs(workspace_dir / "include", exist_ok=True)
    generate_c_interface_header(lowered.libmod_name, [input_tensor], ["output"], workspace_dir / "include")

    input =  np.array([1.0], dtype=input_dtype)
    output = np.zeros(shape=(1,), dtype="float32")
    _create_header_file("input_data", input, workspace_dir / "include")
    _create_header_file("output_data", output, workspace_dir / "include")



    project.build()
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



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


def _create_header_file(output_path, input, output):
    c_types = {
        'int8': 'int8_t',
        'int32': 'int32_t',
        'uint8': 'uint8_t',
        'float32': 'float'
    }

    input_tensor, input_shape, input_dtype = input
    output_tensor, output_shape, output_dtype = output

    file_path = pathlib.Path(f"{output_path}/model_data.h").resolve()

    with open(file_path, "w") as header_file:
        header_file.write(
            "#include <stddef.h>\n"\
            "#include <stdint.h>\n"\
            "#include <dlpack/dlpack.h>\n\n"\
            f"#define model_input_0 {input_tensor}\n"\
            f"#define INPUT_DATA_LEN {np.prod(input_shape)}\n\n"\
            f"{c_types[input_dtype]} input_data[INPUT_DATA_LEN] = {{0}};\n\n"\
            f"#define OUTPUT_DATA_LEN {np.prod(output_shape)}\n"\
            f"{c_types[output_dtype]} output_data[OUTPUT_DATA_LEN] = {{0}};\n\n"
        )


def _read_line(fd, timeout_sec: int):
    data = ""
    new_line = False
    while True:
        if new_line:
            break
        new_data = fd.read(1, timeout_sec=timeout_sec)
        # logging.debug(f"read data: {new_data}")
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
    workspace_dir = create_workspace_dir(platform, os.path.splitext(__file__)[0], mkdir=False)
    model, zephyr_board = PLATFORMS[platform]

    sine_model_path = download_sine_model()
    relay_mod, params, input, output = open_tflite_model(sine_model_path)

    input_tensor, input_shape, input_dtype = input
    output = 'output', output[1], output[2] # TODO check TVM's code generation of default_lib0.c
    output_tensor, output_shape, output_dtype = output

    input_data =  np.zeros(shape=input_shape, dtype=input_dtype)
    output_data = np.zeros(shape=output_shape, dtype=output_dtype)

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

    generated_include_path = workspace_dir / "include"
    os.makedirs(generated_include_path, exist_ok=True)
    generate_c_interface_header(lowered.libmod_name, [input_tensor], ["output"], generated_include_path)
    # _create_header_file(generated_include_path, input, output)

    project.build()
    # project.flash()

    # with project.transport() as transport:
    #     for x in np.arange(0, 2, 0.1):
    #         transport.write(bytes(f"#input:{x}\n", 'UTF-8'), timeout_sec=5)
    #         result_line = _get_message(transport, "#result", timeout_sec=5)
    #         r = result_line.strip("\n").split(":")
    #         logging.info(f"sin({x:.2f}) = {r[1]:.3f}     time: {r[2]} us")

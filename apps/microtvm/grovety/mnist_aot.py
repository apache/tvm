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


def open_mnist8_model(data_dir: str):
    import onnx
    onnx_model = onnx.load(data_dir / "mnist-8.onnx")
    shape = {"Input3": (1, 1, 28, 28)}
    relay_mod, params = relay.frontend.from_onnx(onnx_model, shape=shape, freeze_params=True)
    relay_mod = relay.transform.DynamicToStatic()(relay_mod)

    input = ("Input3", (1, 1, 28, 28), 'float32')
    output = ("Plus214_Output_0", (1, 10), 'float32')
    return (relay_mod, params, input, output)


if __name__ == "__main__":
    workspace_dir = create_workspace_dir(platform, os.path.splitext(__file__)[0], mkdir=False)
    model, zephyr_board = PLATFORMS[platform]

    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    data_dir = current_dir / "mnist_data"
    relay_mod, params, input, output = open_mnist8_model(data_dir)

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
    create_header_file(generated_include_path, input, output)

    project.build()

    from PIL import Image
    digit_2 = Image.open(data_dir / "digit-2.jpg").resize((28, 28))
    digit_2 = np.asarray(digit_2).astype("float32")
    digit_2 = np.expand_dims(digit_2, axis=0)

    digit_9 = Image.open(data_dir / "digit-9.jpg").resize((28, 28))
    digit_9 = np.asarray(digit_9).astype("float32")
    digit_9 = np.expand_dims(digit_9, axis=0)


    # project.flash()

    # with project.transport() as transport:
    #     for x in np.arange(0, 2, 0.1):
    #         transport.write(bytes(f"#input:{x}\n", 'UTF-8'), timeout_sec=5)
    #         result_line = get_message(transport, "#result", timeout_sec=5)
    #         r = result_line.strip("\n").split(":")
    #         logging.info(f"sin({x:.2f}) = {r[1]:.3f}     time: {r[2]} us")

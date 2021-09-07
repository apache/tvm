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

import cifar
import mnist

_LOG = logging.getLogger(__name__)

logging.basicConfig(level="INFO")

TEMPLATE_PROJECT_DIR = tvm_repo_root() + "/apps/microtvm/grovety/template_project"

verbose = True
platform = "stm32f746xx_nucleo"




# from tvm.relay.op import op as reg
# from tvm.relay.qnn.op import layout_conversions

# @reg.register_convert_op_layout("qnn.conv2d", level=11)
# def convert_conv2d(attrs, inputs, tinfos, desired_layouts):
#     channels = tinfos[0].shape[attrs["data_layout"].find("C")]


def apply_Ilya_hack(relay_mod):
    desired_layouts = {'qnn.conv2d': ['NHWC', 'HWOI'], 'nn.conv2d': ['NHWC', 'HWOI']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        return seq(relay_mod)


if __name__ == "__main__":
    workspace_dir = create_workspace_dir(platform, os.path.splitext(__file__)[0], mkdir=False)
    model, zephyr_board = PLATFORMS[platform]

    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    data_dir = current_dir / "mnist"
    relay_mod, params, input, output = mnist.open_Q_model(data_dir / "mnist_model_quant.tflite")

    # relay_mod, params, input, output = cifar.open_model(current_dir / "cifar" / "cifar_quant_8bit.tflite")

    relay_mod = apply_Ilya_hack(relay_mod)


    input_tensor, input_shape, input_dtype = input
    output = 'output', output[1], output[2] # TODO check TVM's code generation of default_lib0.c
    output_tensor, output_shape, output_dtype = output

    input_data =  np.zeros(shape=input_shape, dtype=input_dtype)
    output_data = np.zeros(shape=output_shape, dtype=output_dtype)

    target = tvm.target.target.Target('c -keys=arm_cpu -mcpu=cortex-m7 -march=armv7e-m -model=stm32f746xx -runtime=c -link-params=1 --executor=aot --unpacked-api=1 --interface-api=c')

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
        }
    )

    generated_include_path = workspace_dir / "include"
    os.makedirs(generated_include_path, exist_ok=True)
    generate_c_interface_header(lowered.libmod_name, [input_tensor], ["output"], generated_include_path)
    create_header_file(generated_include_path, input, output)

    project.build()
    project.flash()


    with project.transport() as transport:
        from PIL import Image
        image_files = ["digit-2.jpg", "digit-9.jpg"]

        for file in image_files:
            img = Image.open(data_dir / file).resize((28, 28))
            img = np.asarray(img).astype("uint8")
            # img = np.random.randint(0, high=255, size=input_shape, dtype=input_dtype)
            img = np.reshape(img, -1)
            image_s = ','.join(str(e) for e in img)

            transport.write(bytes(f"#input:{image_s}\n", 'UTF-8'), timeout_sec=5)
            result_line = get_message(transport, "#result", timeout_sec=5)
            r = result_line.strip("\n").split(":")
            output_values = list(map(float, r[1].split(',')))
            max_index = np.argmax(output_values)
            elapsed = int(r[2]) / 1000.0

            logging.info(f"result is {max_index}; time: {elapsed}ms; output: {output_values}")

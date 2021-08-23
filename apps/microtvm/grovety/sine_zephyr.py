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
import logging
import os
import pathlib
import logging
import numpy as np
import tvm
from tvm.micro.transport import Transport
import tvm.rpc
import tvm.micro
import tvm.testing
import tvm.relay as relay
from tvm.contrib.download import download_testdata

_LOG = logging.getLogger(__name__)

PLATFORMS = {
    "qemu_x86": ("host", "qemu_x86"),
    "qemu_riscv32": ("host", "qemu_riscv32"),
    "qemu_riscv64": ("host", "qemu_riscv64"),
    "mps2_an521": ("mps2_an521", "mps2_an521"),
    "nrf5340dk": ("nrf5340dk", "nrf5340dk_nrf5340_cpuapp"),
    "stm32f746xx_disco": ("stm32f746xx", "stm32f746g_disco"),
    "stm32f746xx_nucleo": ("stm32f746xx", "nucleo_f746zg"),
    "stm32l4r5zi_nucleo": ("stm32l4r5zi", "nucleo_l4r5zi"),
    "zynq_mp_r5": ("zynq_mp_r5", "qemu_cortex_r5"),
}

logging.basicConfig(level="INFO")

TEMPLATE_PROJECT_DIR = "/home/sergei/projects/MIR/TVM/tvm/apps/microtvm/zephyr/template_project"

verbose = False
platform = "stm32f746xx_nucleo"


def create_workspace_dir(platform):
    _, zephyr_board = PLATFORMS[platform]
    parent_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    filename = os.path.splitext(os.path.basename(__file__))[0]
    board_workspace = (
        parent_dir
        / f"workspace_{filename}_{zephyr_board}"
        / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )
    board_workspace_base = str(board_workspace)
    number = 1
    while board_workspace.exists():
        board_workspace = pathlib.Path(board_workspace_base + f"-{number}")
        number += 1

    if not os.path.exists(board_workspace.parent):
        os.makedirs(board_workspace.parent)

    os.makedirs(board_workspace)
    return board_workspace

def open_sine_model():
    model_url = "https://people.linaro.org/~tom.gall/sine_model.tflite"
    model_file = "sine_model.tflite"
    model_path = download_testdata(model_url, model_file, module="data")

    tflite_model_buf = open(model_path, "rb").read()

    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)


    input_tensor = "dense_4_input"
    input_shape = (1,)
    input_dtype = "float32"
    input = (input_tensor, input_shape, input_dtype)

    relay_mod, params = relay.frontend.from_tflite(tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype})

    return (relay_mod, params, input)


if __name__ == '__main__':
    workspace_dir = create_workspace_dir(platform)
    model, zephyr_board = PLATFORMS[platform]

    relay_mod, params, input = open_sine_model()
    input_tensor, input_shape, input_dtype = input

    relay_mod = relay.transform.DynamicToStatic()(relay_mod)


    target = tvm.target.target.micro(model, options=["-link-params=1"])
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params)
        graph = lowered.get_graph_json()


    project = tvm.micro.generate_project(
        str(TEMPLATE_PROJECT_DIR),
        lowered,
        workspace_dir / "project",
        {
            "project_type": "host_driven",
            "west_cmd": "west",
            "verbose": verbose,
            "zephyr_board": zephyr_board,
        },
    )
    project.build()
    project.flash()

    with tvm.micro.Session(project.transport()) as session:
        graph_mod = tvm.micro.create_local_graph_executor(graph, session.get_system_lib(), session.device)

        for index in range(20):
            x = index * 0.1

            graph_mod.set_input(input_tensor, tvm.nd.array(np.array([x], dtype=input_dtype)))
            graph_mod.run()
            y = graph_mod.get_output(0).numpy()

            print(f"sin({x:.2f}) = {y}")


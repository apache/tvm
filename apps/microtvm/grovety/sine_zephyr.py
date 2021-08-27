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
import logging
import numpy as np
import tvm
import tvm.rpc
import tvm.micro
import tvm.testing
import tvm.relay as relay
from common import *

_LOG = logging.getLogger(__name__)


logging.basicConfig(level="INFO")

TEMPLATE_PROJECT_DIR = tvm_repo_root() + "/apps/microtvm/grovety/template_project"

verbose = False
platform = "stm32f746xx_disco"
# platform = "LPCXpresso5569"
# platform = "stm32f746xx_nucleo"

if __name__ == '__main__':
    workspace_dir = create_workspace_dir(platform, 'sine_zephyr', mkdir=True)
    target, zephyr_board = PLATFORMS[platform]

    sine_model_path = download_sine_model()


    relay_mod, params, input = open_tflite_model(sine_model_path)
    input_tensor, input_shape, input_dtype = input

    print_relay(relay_mod, params)

    relay_mod = relay.transform.DynamicToStatic()(relay_mod)


    target = tvm.target.target.micro(target, options=["-link-params=1"])
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


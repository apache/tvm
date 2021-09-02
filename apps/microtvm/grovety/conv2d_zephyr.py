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
from tf_common import *

_LOG = logging.getLogger(__name__)


logging.basicConfig(level="INFO")

TEMPLATE_PROJECT_DIR = tvm_repo_root() + "/apps/microtvm/zephyr/template_project"

verbose = False
platform = "stm32f746xx_disco"

test_cases = [
    # input_shape, weights, dilations, strides, padding, format, input_dtype
    ((1, 24, 24, 4), (3, 3, 4, 4), [1, 1, 1, 1], [1, 3, 3, 1], "SAME", "NHWC", "int32"),
]

def get_tf_results(input_data, input_node, output_node):
    import tensorflow as tf

    in_data, w_data, padding, format, input_dtype = input_data
    # prepare net
    tf.compat.v1.reset_default_graph()
    input = tf.compat.v1.placeholder(input_dtype, name=input_node, shape=[None] + list(input_shape[1:]))
    filter = tf.constant(w_data, dtype=input_dtype)

    # operation for test
    tf.nn.conv2d(
        input=input,
        filters=filter,
        strides=strides,
        padding=padding,
        data_format=format,
        dilations=dilations,
        name=output_node
    )

    # run tf
    tf_val = run_tf(in_data, pb_file, input_node, output_node)

    return pb_file, tf_val

if __name__ == '__main__':
    workspace_dir = create_workspace_dir(platform, 'conv2d_zephyr', mkdir=True)
    model, zephyr_board = PLATFORMS[platform]

    # prepare settings
    for case in test_cases:
        input_shape, weights, dilations, strides, padding, format, input_dtype = case

        pb_file = os.path.join(workspace_dir, "{}.pb".format(os.path.basename(__file__)))

        input_tensor = "input_node"
        output_tensor = "output_node"

        # prepare data
        in_data, w_data = get_values(input_shape, input_dtype, weights)

        pb_file, tf_val = get_tf_results([in_data, w_data, padding, format, input_dtype], input_tensor, output_tensor)
        relay_mod, params = open_conv2d_model(pb_file, input_tensor, input_shape, format)

        target = tvm.target.target.micro(model, options=["-keys=arm_cpu,cpu -link-params=1"])
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

            tvm_in_data = np.array(in_data)[np.newaxis, :]

            graph_mod.set_input(input_tensor, tvm.nd.array(tvm_in_data.astype(input_dtype)))
            graph_mod.run()
            tvm_val = graph_mod.get_output(0).numpy()

            tvm.testing.assert_allclose(tf_val, tvm_val, rtol=1e-3, atol=1e-7)
            print(tf_val-tvm_val)
        print("Done")

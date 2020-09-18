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
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from test_quantize import (
    create_conv2d_bias_func,
    create_q_conv2d_bias_func,
)

from tvm.relay.transform.quantize import (
    QuantizePass,
    all_patterns,
    average_max_per_channel_patterns,
    GlobalCalibrationCallback,
)

import numpy as np


def verify_pass(pre_mod, params, input_dict, quantizer_pattern_list):
    opt = QuantizePass(quantizer_pattern_list, params, skip_first=False)
    with relay.build_config(opt_level=3):
        post_mod = opt(pre_mod)
        q_lib = relay.build(post_mod, params=params, target="llvm")

    q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
    q_gmod.set_input(**input_dict)
    q_gmod.run()


def create_conv2d_bias_mods(data_shape, weight_shape, bias_shape, attrs, bias_type="bias_add"):
    pre_func, data, weight, bias = create_conv2d_bias_func(
        data_shape, weight_shape, bias_shape, attrs, bias_type
    )
    pre_mod = tvm.IRModule.from_expr(pre_func)
    params = {"weight": np.random.randn(*weight_shape).astype("float32")}
    input_dict = {"data": np.random.randn(*data_shape).astype("float32")}
    return pre_mod, params, input_dict


def test_pass():
    cc = GlobalCalibrationCallback(0.05, 0.1)

    pre_mod, params, input_dict = create_conv2d_bias_mods(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        (32,),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
    )
    verify_pass(pre_mod, params, input_dict, all_patterns(cc))


if __name__ == "__main__":
    test_pass()

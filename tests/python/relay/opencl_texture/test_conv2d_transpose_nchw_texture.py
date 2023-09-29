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

import re
import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
from utils.adreno_utils import gpu_preprocess, build_run_compare, build_run_compare_vm
import pytest

executor_type = tvm.testing.parameter("ge")
dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_transpose_adreno(remote, target, executor_type, dtype):

    trials = [
        [4, 4, (1, 1), (2, 2), (1, 1), 64, (256, 100, 100), (False, False)],
        [4, 4, (0, 0), (2, 2), (1, 1), 256, (32, 64, 64), (False, False)],
        [3, 3, (0, 0), (2, 2), (1, 1), 64, (256, 12, 12), (True, True)],
        [4, 4, (1, 1), (1, 1), (1, 1), 512, (16, 100, 100), (False, False)],
        [5, 5, (2, 2), (2, 2), (1, 1), 4, (16, 100, 100), (True, False)],
        [7, 7, (3, 3), (2, 2), (1, 1), 8, (4, 100, 100), (False, True)],
    ]

    for (
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        out_channels,
        shape,
        composite,
    ) in trials:
        shape = (1, *shape)
        has_bias = composite[0]
        has_activation = composite[1]
        input_shape = shape
        filter_shape = (shape[1], out_channels, kernel_w, kernel_h)
        x = relay.var("data", shape=input_shape, dtype=dtype)
        w = relay.var("weight", shape=filter_shape, dtype=dtype)
        inputs = [x, w]
        y = relay.nn.conv2d_transpose(
            x,
            w,
            channels=out_channels,
            kernel_size=(kernel_w, kernel_h),
            strides=stride,
            padding=pad,
            kernel_layout="IOHW",
            data_layout="NCHW",
            dilation=dilation,
        )
        np.random.seed(0)
        initializer = relay.testing.init.Xavier()
        filter_data = np.zeros(filter_shape).astype(dtype)
        initializer("weight", filter_data)
        params1 = {
            "weight": tvm.nd.array(filter_data),
        }

        if has_bias:
            b = relay.var("bias", shape=(out_channels,), dtype=dtype)
            y = relay.nn.bias_add(y, b, axis=1)
            inputs.append(b)
            bias_data = np.zeros((out_channels,)).astype(dtype)
            initializer("bias", bias_data)
            params1["bias"] = tvm.nd.array(bias_data)

        if has_activation:
            y = relay.nn.relu(y)

        mod = relay.Function(inputs, y)

        if executor_type == "ge":
            build_run_compare(
                remote,
                mod,
                params1,
                {"data": input_shape},
                {"data": dtype},
                target,
                [],
                gpu_preprocess,
            )
        else:
            build_run_compare_vm(
                remote,
                mod,
                params1,
                {"data": input_shape},
                {"data": dtype},
                target,
                [],
                gpu_preprocess,
            )


if __name__ == "__main__":
    tvm.testing.main()

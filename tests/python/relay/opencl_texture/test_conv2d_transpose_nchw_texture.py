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


executor_type = tvm.testing.parameter("ge", "vm")
dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_transpose_adreno(remote, target, executor_type, dtype):
    # Conv2d transpose test cases lists
    trials = [
        [4, 4, (1, 1), (2, 2), (1, 1), 64, (256, 100, 100), (False, False), gpu_preprocess],
        [4, 4, (0, 0), (2, 2), (1, 1), 256, (32, 64, 64), (False, False), None],
        [3, 3, (0, 0), (2, 2), (1, 1), 64, (256, 100, 100), (True, True), None],
        [4, 4, (1, 1), (1, 1), (1, 1), 512, (16, 100, 100), (False, False), gpu_preprocess],
        [5, 5, (2, 2), (2, 2), (1, 1), 4, (16, 100, 100), (True, False), gpu_preprocess],
        [7, 7, (3, 3), (2, 2), (1, 1), 8, (4, 100, 100), (False, True), None],
        [7, 7, (3, 3), (2, 2), (1, 1), 64, (3, 100, 100), (True, True), None],
        [3, 3, (1, 1), (1, 1), (1, 1), 3, (16, 8, 8), (True, True), None],
    ]
    # Tensors memory scope with graph executor build
    ge_texture_scopes = [
        ["", "global.texture", "global.texture-weight", "", ""],
        ["", "global.texture", "global.texture-weight", "", ""],
        ["", "global.texture", "global.texture-weight", "global.texture-weight", "", ""],
        ["", "global.texture", "global.texture-weight", "", ""],
        ["", "global.texture", "global.texture-weight", "global.texture-weight", "", ""],
        ["", "global.texture", "global.texture-nhwc", "", ""],
        [],
        [],
    ]
    # Tensors memory scope with vm executor build
    vm_texture_scopes = [
        """
        VM VirtualDevice[0]: device type 1, id 0 and mem_scope
        VM VirtualDevice[1]: device type 4, id 0 and mem_scope
        VM VirtualDevice[2]: device type 4, id 0 and mem_scope global.texture
        VM VirtualDevice[3]: device type 4, id 0 and mem_scope global.texture-weight
        """,
        """
        VM VirtualDevice[0]: device type 1, id 0 and mem_scope
        VM VirtualDevice[1]: device type 4, id 0 and mem_scope
        VM VirtualDevice[2]: device type 4, id 0 and mem_scope global.texture
        VM VirtualDevice[3]: device type 4, id 0 and mem_scope global.texture-weight
        """,
        """
        VM VirtualDevice[0]: device type 1, id 0 and mem_scope
        VM VirtualDevice[1]: device type 4, id 0 and mem_scope
        VM VirtualDevice[2]: device type 4, id 0 and mem_scope global.texture
        VM VirtualDevice[3]: device type 4, id 0 and mem_scope global.texture-weight
        VM VirtualDevice[4]: device type 4, id 0 and mem_scope global.texture-weight
        """,
        """
        VM VirtualDevice[0]: device type 1, id 0 and mem_scope
        VM VirtualDevice[1]: device type 4, id 0 and mem_scope
        VM VirtualDevice[2]: device type 4, id 0 and mem_scope global.texture
        VM VirtualDevice[3]: device type 4, id 0 and mem_scope global.texture-weight
        """,
        """
        VM VirtualDevice[0]: device type 1, id 0 and mem_scope
        VM VirtualDevice[1]: device type 4, id 0 and mem_scope
        VM VirtualDevice[2]: device type 4, id 0 and mem_scope global.texture
        VM VirtualDevice[3]: device type 4, id 0 and mem_scope global.texture-weight
        VM VirtualDevice[4]: device type 4, id 0 and mem_scope global.texture-weight
        """,
        """
        VM VirtualDevice[0]: device type 1, id 0 and mem_scope
        VM VirtualDevice[1]: device type 4, id 0 and mem_scope
        VM VirtualDevice[2]: device type 4, id 0 and mem_scope global.texture
        VM VirtualDevice[3]: device type 4, id 0 and mem_scope global.texture-nhwc
        """,
        [],
        [],
    ]

    for i, (
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        out_channels,
        shape,
        composite,
        _gpu_preprocess,
    ) in enumerate(trials):
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
                ge_texture_scopes[i],
                _gpu_preprocess,
            )
        else:
            build_run_compare_vm(
                remote,
                mod,
                params1,
                {"data": input_shape},
                {"data": dtype},
                target,
                vm_texture_scopes[i],
                _gpu_preprocess,
            )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_transpose_three_layer_block(remote, target, executor_type, dtype):
    # Conv2d transpose test cases lists
    trials = [
        [4, 4, (1, 1), (2, 2), (1, 1), 64, (256, 100, 100), (False, False), None],
        [3, 3, (0, 0), (1, 1), (1, 1), 64, (256, 12, 12), (True, True), gpu_preprocess],
    ]
    ge_texture_scopes = [
        [
            "",
            "global.texture",
            "global.texture-weight",
            "global.texture",
            "global.texture-weight",
            "global.texture",
            "global.texture-weight",
            "",
            "",
        ],
        [
            "",
            "global.texture-nhwc",
            "global.texture-weight",
            "global.texture-nhwc",
            "global.texture-weight",
            "global.texture-weight",
            "global.texture-nhwc",
            "global.texture-weight",
            "",
            "",
        ],
    ]
    vm_texture_scopes = [
        """
        VM VirtualDevice[0]: device type 1, id 0 and mem_scope
        VM VirtualDevice[1]: device type 4, id 0 and mem_scope
        VM VirtualDevice[2]: device type 4, id 0 and mem_scope global.texture
        VM VirtualDevice[3]: device type 4, id 0 and mem_scope global.texture
        VM VirtualDevice[4]: device type 4, id 0 and mem_scope global.texture-weight
        VM VirtualDevice[5]: device type 4, id 0 and mem_scope global.texture
        VM VirtualDevice[6]: device type 4, id 0 and mem_scope global.texture-weight
        VM VirtualDevice[7]: device type 4, id 0 and mem_scope global.texture-weight
        """,
        """
        VM VirtualDevice[0]: device type 1, id 0 and mem_scope
        VM VirtualDevice[1]: device type 4, id 0 and mem_scope
        VM VirtualDevice[2]: device type 4, id 0 and mem_scope global.texture-nhwc
        VM VirtualDevice[3]: device type 4, id 0 and mem_scope global.texture-nhwc
        VM VirtualDevice[4]: device type 4, id 0 and mem_scope global.texture-weight
        VM VirtualDevice[5]: device type 4, id 0 and mem_scope global.texture-nhwc
        VM VirtualDevice[6]: device type 4, id 0 and mem_scope global.texture-weight
        VM VirtualDevice[7]: device type 4, id 0 and mem_scope global.texture-weight
        VM VirtualDevice[8]: device type 4, id 0 and mem_scope global.texture-weight
        """,
    ]

    for i, (
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        out_channels,
        shape,
        composite,
        _gpu_preprocess,
    ) in enumerate(trials):
        shape = (1, *shape)
        has_bias = composite[0]
        has_activation = composite[1]
        input_shape = shape
        filter_shape = (shape[1], out_channels, kernel_w, kernel_h)
        x = relay.var("data", shape=input_shape, dtype=dtype)
        w = relay.var("weight", shape=filter_shape, dtype=dtype)
        inputs = [x, w]
        W1 = relay.var("weight1", shape=(shape[1], shape[1], 1, 1), dtype=dtype)
        conv = relay.nn.conv2d(x, W1, padding=[0, 0, 0, 0], channels=shape[1], kernel_size=(1, 1))
        inputs.append(W1)
        conv = relay.op.nn.relu(conv)
        y = relay.nn.conv2d_transpose(
            conv,
            w,
            channels=out_channels,
            kernel_size=(kernel_w, kernel_h),
            strides=stride,
            padding=pad,
            kernel_layout="IOHW",
            data_layout="NCHW",
            dilation=dilation,
        )

        if has_bias:
            b = relay.var("bias", shape=(out_channels,), dtype=dtype)
            y = relay.nn.bias_add(y, b, axis=1)
            inputs.append(b)

        if has_activation:
            y = relay.nn.relu(y)
        W2 = relay.var("weight2", shape=(out_channels, out_channels, 1, 1), dtype=dtype)
        out = relay.nn.conv2d(
            y, W2, padding=[0, 0, 0, 0], channels=out_channels, kernel_size=(1, 1)
        )
        out = relay.op.nn.relu(out)
        np.random.seed(0)
        inputs.append(W2)
        initializer = relay.testing.init.Xavier()
        filter_data = np.zeros(filter_shape).astype(dtype)
        initializer("weight", filter_data)
        filter_data1 = np.zeros((shape[1], shape[1], 1, 1)).astype(dtype)
        initializer("weight", filter_data1)
        filter_data2 = np.zeros((out_channels, out_channels, 1, 1)).astype(dtype)
        initializer("weight", filter_data2)
        params1 = {
            "weight": tvm.nd.array(filter_data),
            "weight1": tvm.nd.array(filter_data1),
            "weight2": tvm.nd.array(filter_data2),
        }
        if has_bias:
            bias_data = np.zeros((out_channels,)).astype(dtype)
            initializer("bias", bias_data)
            params1["bias"] = tvm.nd.array(bias_data)

        mod = relay.Function(inputs, out)

        if executor_type == "ge":
            build_run_compare(
                remote,
                mod,
                params1,
                {"data": input_shape},
                {"data": dtype},
                target,
                ge_texture_scopes[i],
                _gpu_preprocess,
            )
        else:
            build_run_compare_vm(
                remote,
                mod,
                params1,
                {"data": input_shape},
                {"data": dtype},
                target,
                vm_texture_scopes[i],
                _gpu_preprocess,
            )


if __name__ == "__main__":
    tvm.testing.main()

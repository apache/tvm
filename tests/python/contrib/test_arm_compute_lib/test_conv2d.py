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
"""Arm Compute Library integration conv2d tests."""

import numpy as np

import tvm
from tvm import relay

from .infrastructure import skip_runtime_test, skip_codegen_test, build_and_run, \
    verify, verify_codegen
from .infrastructure import Device


def _get_model(shape, kernel_size, padding, strides,
               dilation, groups, dtype, channels,
               var_names, has_bias=False, has_activation=False, has_pad=False):
    """Return a model and any parameters it may have"""
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    if has_pad:
        p = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0))
        a = relay.nn.pad(a, pad_width=p)
        padding = (0, 0, 0, 0)
    else:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        shape = (shape[0], shape[1] + padding[0] * 2,
                 shape[2] + padding[1] * 2, shape[3])
    weight_shape = (kernel_size, kernel_size, shape[3] // groups, channels)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.conv2d(
        a,
        weights,
        kernel_size=(kernel_size, kernel_size),
        data_layout="NHWC",
        kernel_layout="HWIO",
        dilation=(1, 1),
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels
    )
    params = {"w": w}
    if has_bias:
        b = tvm.nd.array(np.random.uniform(-128, 127, weight_shape[3]).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.nn.bias_add(out, biasc, axis=3)
        params["b"] = b
    if has_activation:
        out = relay.nn.relu(out)
    return out, params


def _get_expected_codegen(shape, kernel_size, padding, strides,
                          dilation, groups, dtype, channels,
                          has_bias=False, has_activation=False):
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    weight_shape = (channels, kernel_size, kernel_size, shape[3] // groups)
    output_height = ((shape[1] - kernel_size + padding[0] + padding[2]) / strides[0]) + 1
    output_width = ((shape[2] - kernel_size + padding[1] + padding[3]) / strides[1]) + 1
    output_shape = (1, int(output_height), int(output_width), channels)

    node = {
            "op": "kernel",
            "name": "nn.conv2d",
            "inputs": [[0, 0, 0], [1, 0, 0]],
            "attrs": {
                "groups": [["1"]],
                "num_inputs": str(3 if has_bias else 2),
                "num_outputs": "1",
                "data_layout": [["NHWC"]],
                "kernel_layout": [["OHWI"]],
                "channels": [["1"]],
                "dilation": [["1", "1"]],
                "out_layout": [[""]],
                "out_dtype": [[""]],
                "kernel_size": [[str(kernel_size), str(kernel_size)]],
                "shape": [[list(output_shape)]],
                "dtype": [[dtype]],
                "padding": [[str(p) for p in padding]],
                "strides": [[str(s) for s in strides]]
            },
        }

    if has_activation:
        node["attrs"]["activation_type"] = [["relu"]]

    input = {
        "op": "input",
        "name": "",
        "attrs": {"shape": [[list(shape)]], "dtype": [["float32"]]}}
    kernel = {
        "op": "const",
        "name": "",
        "attrs": {"shape": [[list(weight_shape)]], "dtype": [["float32"]]}}

    if has_bias:
        bias = {
            "op": "const",
            "name": "",
            "attrs": {"shape": [[[weight_shape[0]]]], "dtype": [["float32"]]}}
        node["inputs"].append([2, 0, 0])
        return [input, kernel, bias, node]
    else:
        return [input, kernel, node]


def test_conv2d():
    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    shape = (1, 14, 14, 32)
    dtype = "float32"

    inputs = {
        "a": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype)),
    }

    for kernel_size in [1, 2, 3]:
        outputs = []
        func, params = _get_model(shape, kernel_size,
                                  (0, 0), (1, 1), 1, 1,
                                  dtype, 1, iter(inputs))
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1,
                                         params, device,
                                         enable_acl=acl)[0])
        verify(outputs, atol=0.002, rtol=0.01)

    for pad_ksize in [((1, 1), 3), ((2, 2), 5), ((2, 1), 3)]:
        outputs = []
        func, params = _get_model(shape, pad_ksize[1], pad_ksize[0],
                                  (1, 1), 1, 1, dtype, 1, iter(inputs))
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1,
                                         params, device,
                                         enable_acl=acl)[0])
        verify(outputs, atol=0.002, rtol=0.01)

    for strides in [(1, 1), (2, 2)]:
        outputs = []
        func, params = _get_model(shape, 2, (0, 0), strides,
                                  1, 1, dtype, 1, iter(inputs))
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1,
                                         params, device,
                                         enable_acl=acl)[0])
        verify(outputs, atol=0.002, rtol=0.01)

    # Test composite convolution: (has_pad, has_bias, has_activation).
    for composite in [(False, True, False), (False, False, True), (False, True, True),
                      (True, False, False)]:
        outputs = []
        func, params = _get_model(shape, 2, (1, 1), (1, 1),
                                  1, 1, dtype, 1, iter(inputs),
                                  has_pad=composite[0],
                                  has_bias=composite[1],
                                  has_activation=composite[2])
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1,
                                         params, device,
                                         enable_acl=acl)[0])
        verify(outputs, atol=0.002, rtol=0.01)


def test_codegen_conv2d():
    if skip_codegen_test():
        return

    shape = (1, 25, 25, 1)
    dtype = "float32"
    inputs = {"a"}

    for pad_ksize in [((1, 1), 3), ((2, 1), 3)]:
        args = (shape, pad_ksize[1], pad_ksize[0], (1, 1), 1, 1, dtype, 1)
        func, params = _get_model(*args, var_names=iter(inputs))
        exp_codegen = _get_expected_codegen(*args)
        verify_codegen(func, exp_codegen, 1)
    # Test composite convolution: (has_pad, has_bias, has_activation).
    for composite in [(False, True, False), (False, False, True), (False, True, True),
                      (True, False, False)]:
        args = (shape, 2, (1, 1), (1, 1), 1, 1, dtype, 1)
        func, params = _get_model(*args, var_names=iter(inputs),
                                  has_pad=composite[0],
                                  has_bias=composite[1],
                                  has_activation=composite[2])
        exp_codegen = _get_expected_codegen(*args,
                                            has_bias=composite[1],
                                            has_activation=composite[2])
        verify_codegen(func, exp_codegen, 1)


if __name__ == "__main__":
    test_conv2d()
    test_codegen_conv2d()

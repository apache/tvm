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
"""ACL Integration conv2d tests."""

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
    codegen = {
        "name": "conv2d",
        "inputs": [],
        "outputs": [],
        "attrs": {
            "groups": ["Int", 1],
            "num_inputs": ["Size_t", 2],
            "num_outputs": ["Size_t", 1]
        }
    }

    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    # Transpose padding to match ACL format
    padding = (padding[1], padding[3], padding[0], padding[2])
    weight_shape = (channels, kernel_size, kernel_size, shape[3] // groups)
    output_height = ((shape[1] - kernel_size + padding[2] + padding[3]) / strides[0]) + 1
    output_width = ((shape[2] - kernel_size + padding[0] + padding[1]) / strides[1]) + 1
    output_shape = (1, int(output_height), int(output_width), channels)

    codegen["attrs"]["padding"] = ["IntVector", list(padding)]
    codegen["attrs"]["strides"] = ["IntVector", list(strides)]
    if has_activation:
        codegen["attrs"]["activation_type"] = ["String", "relu"]

    inputs = [{"type": "var", "shape": list(shape)},
              {"type": "const", "shape": list(weight_shape)}]
    if has_bias:
        inputs.append({"type": "const", "shape": [weight_shape[0]]})
    outputs = [{"type": "var", "shape": list(output_shape)}]

    codegen["inputs"] = inputs
    codegen["outputs"] = outputs
    codegen["attrs"]["num_inputs"] = ["Size_t", len(inputs)]
    codegen["attrs"]["num_outputs"] = ["Size_t", len(outputs)]

    return codegen


def test_conv2d():
    if skip_runtime_test():
        return

    device = Device()

    shape = (1, 25, 25, 1)
    dtype = "float32"

    inputs = {
        "a": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype)),
    }

    for kernel_size in [2, 3]:
        outputs = []
        func, params = _get_model(shape, kernel_size,
                                  (0, 0), (1, 1), 1, 1,
                                  dtype, 1, iter(inputs))
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1,
                                         params, device,
                                         enable_acl=acl))
        verify(outputs, atol=0.002, rtol=0.01)

    for pad_ksize in [((1, 1), 3), ((2, 2), 5), ((2, 1), 3)]:
        outputs = []
        func, params = _get_model(shape, pad_ksize[1], pad_ksize[0],
                                  (1, 1), 1, 1, dtype, 1, iter(inputs))
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1,
                                         params, device,
                                         enable_acl=acl))
        verify(outputs, atol=0.002, rtol=0.01)

    for strides in [(1, 1), (2, 2)]:
        outputs = []
        func, params = _get_model(shape, 2, (0, 0), strides,
                                  1, 1, dtype, 1, iter(inputs))
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1,
                                         params, device,
                                         enable_acl=acl))
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
                                         enable_acl=acl))
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
                                            has_activation=composite[2],
                                            )
        verify_codegen(func, exp_codegen, 1)


if __name__ == "__main__":
    test_conv2d()
    test_codegen_conv2d()

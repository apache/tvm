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
"""Arm Compute Library integration pooling tests."""

import numpy as np

import tvm
from tvm import relay

from .infrastructure import skip_runtime_test, skip_codegen_test, build_and_run, \
    verify, verify_codegen
from .infrastructure import Device


def _get_model(shape, dtype, typef, sizes, strides, padding,
               ceil_mode, var_names):
    """Return a model and any parameters it may have."""
    var = relay.var(next(var_names), shape=shape, dtype=dtype)
    pool = typef(var, pool_size=sizes, strides=strides, padding=padding,
                 ceil_mode=ceil_mode, layout="NHWC")
    return pool


def _get_expected_codegen(shape, dtype, typef, sizes, strides,
                          padding, ceil_mode):
    if len(padding) == 2:
        padding = (padding[1], padding[1], padding[0], padding[0])
    output_height = ((shape[1] - sizes[0] + padding[0] + padding[2]) / strides[0]) + 1
    output_width = ((shape[2] - sizes[1] + padding[1] + padding[3]) / strides[1]) + 1
    output_shape = (1, int(output_height), int(output_width), shape[3])

    node = {
        "op": "kernel",
        "name": "nn.max_pool2d",
        "inputs": [[0, 0, 0]],
        "attrs": {
            "num_inputs": "1",
            "num_outputs": "1",
            "layout": [["NHWC"]],
            "shape": [[list(output_shape)]],
            "dtype": [[dtype]],
            "padding": [[str(p) for p in padding]],
            "strides": [[str(s) for s in strides]],
            "pool_size": [[str(s) for s in sizes]],
            "ceil_mode": [[str(1 if ceil_mode else 0)]]
        },
    }

    input = {
        "op": "input",
        "name": "",
        "attrs": {"shape": [[list(shape)]], "dtype": [[dtype]]}}
    return [input, node]


def test_pooling():
    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    for dtype, low, high, atol, rtol in [("float32", -127, 128, 0.001, 0.001), ("uint8", 0, 255, 0, 0)]:
        for size in [(2, 2), (3, 3)]:
            for stride in [(2, 2)]:
                shape = (1, size[0] + stride[0] * 5,
                         size[1] + stride[1] * 5, 16)
                pad = (0, 0)

                inputs = {
                    "a": tvm.nd.array(np.random.uniform(low, high, shape).astype(dtype)),
                }

                outputs = []
                func = _get_model(shape, dtype, relay.nn.max_pool2d, size,
                                  stride, pad, True, iter(inputs))
                for acl in [False, True]:
                    outputs.append(build_and_run(func, inputs, 1, None, device,
                                                 enable_acl=acl)[0])

                params = {
                    "size": size,
                    "stride": stride,
                    "shape": shape,
                    "pooling type": "max",
                    "dtype": dtype,
                    "padding": pad
                }
                verify(outputs, atol=atol, rtol=rtol, params=params)


def test_codegen_pooling():
    if skip_codegen_test():
        return

    inputs = {"a"}

    for dtype in ["float32", "uint8"]:
        for size in [(2, 2), (3, 3)]:
            for stride in [(2, 2)]:
                shape = (1, size[0] + stride[0] * 5,
                         size[1] + stride[1] * 5, 16)
                args = (shape, dtype, relay.nn.max_pool2d, size,
                        stride, (0, 0), True)
                func = _get_model(*args, iter(inputs))
                exp_codegen = _get_expected_codegen(*args)
                verify_codegen(func, exp_codegen, 1)


if __name__ == "__main__":
    test_pooling()
    test_codegen_pooling()

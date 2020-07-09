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
"""ACL Integration pooling tests."""

import numpy as np

import tvm
from tvm import relay

from .infrastructure import skip_runtime_test, skip_codegen_test, build_and_run, \
    verify, verify_codegen
from .infrastructure import Device


def _get_model(shape, typef, sizes, strides, padding,
               ceil_mode, var_names):
    """Return a model and any parameters it may have."""
    var = relay.var(next(var_names), shape=shape, dtype="float32")
    pool = typef(var, pool_size=sizes, strides=strides, padding=padding,
                 ceil_mode=ceil_mode, layout="NHWC")
    return pool


def _get_expected_codegen(shape, typef, sizes, strides, padding,
                          ceil_mode):
    codegen = {
        "name": "max_pool",
        "inputs": [],
        "outputs": [],
        "attrs": {
            "pooling_type": ["String", "max"]
        }
    }

    if len(padding) == 2:
        padding = (padding[1], padding[1], padding[0], padding[0])
    # Transpose padding to match ACL format
    padding = (padding[1], padding[3], padding[0], padding[2])
    output_height = ((shape[1] - sizes[0] + padding[2] + padding[3]) / strides[0]) + 1
    output_width = ((shape[2] - sizes[1] + padding[0] + padding[1]) / strides[1]) + 1
    output_shape = (1, int(output_height), int(output_width), shape[3])

    if typef == relay.nn.max_pool2d:
        pooling_type = "max"
    else:
        raise NotImplementedError(f"No conversion from {typef} to pooling_type string.")

    codegen["attrs"]["padding"] = ["IntVector", list(padding)]
    codegen["attrs"]["strides"] = ["IntVector", list(strides)]
    codegen["attrs"]["pool_size"] = ["IntVector", list(sizes)]
    codegen["attrs"]["pooling_type"] = ["String", pooling_type]

    inputs = [{"type": "var", "shape": list(shape)}]
    outputs = [{"type": "var", "shape": list(output_shape)}]

    codegen["inputs"] = inputs
    codegen["outputs"] = outputs
    codegen["attrs"]["num_inputs"] = ["Size_t", len(inputs)]
    codegen["attrs"]["num_outputs"] = ["Size_t", len(outputs)]

    return codegen


def test_pooling():
    if skip_runtime_test():
        return

    device = Device()

    for size in [(2, 2), (3, 3)]:
        for stride in [(2, 2)]:
            shape = (1, size[0] + stride[0] * 5,
                     size[1] + stride[1] * 5, 16)

            inputs = {
                "a": tvm.nd.array(np.random.uniform(-1, 1, shape).astype("float32")),
            }

            outputs = []
            func = _get_model(shape, relay.nn.max_pool2d, size,
                              stride, (0, 0), True, iter(inputs))
            for acl in [False, True]:
                outputs.append(build_and_run(func, inputs, 1, None, device,
                                             enable_acl=acl))
            verify(outputs, atol=0.001, rtol=0.001)


def test_codegen_pooling():
    if skip_codegen_test():
        return

    inputs = {"a"}

    for size in [(2, 2), (3, 3)]:
        for stride in [(2, 2)]:
            shape = (1, size[0] + stride[0] * 5,
                     size[1] + stride[1] * 5, 16)
            args = (shape, relay.nn.max_pool2d, size,
                    stride, (0, 0), True)
            func = _get_model(*args, iter(inputs))
            exp_codegen = _get_expected_codegen(*args)
            verify_codegen(func, exp_codegen, 1)


if __name__ == "__main__":
    test_pooling()
    test_codegen_pooling()

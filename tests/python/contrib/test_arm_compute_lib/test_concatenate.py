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
"""Arm Compute Library integration concatenate tests."""

import numpy as np

import tvm
from tvm import relay
from tvm import testing

from test_arm_compute_lib.infrastructure import (
    skip_runtime_test,
    skip_codegen_test,
    build_and_run,
    verify,
    verify_codegen,
)
from test_arm_compute_lib.infrastructure import Device


def _get_model(input_shape_a, input_shape_b, input_shape_c, axis, dtype, var_names):
    """Return a model and any parameters it may have."""
    a = relay.var(next(var_names), shape=input_shape_a, dtype=dtype)
    b = relay.var(next(var_names), shape=input_shape_b, dtype=dtype)
    c = relay.var(next(var_names), shape=input_shape_c, dtype=dtype)
    out = relay.concatenate([a, b, c], axis)
    return out


def _get_expected_codegen(input_shape_a, input_shape_b, input_shape_c, axis, dtype):
    node = {
        "op": "kernel",
        "name": "concatenate",
        "inputs": [
            [0, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
        ],
        "attrs": {
            "num_outputs": "1",
            "num_inputs": "3",
            "dtype": [[dtype]],
            "axis": [[str(axis)]],
            "shape": [[[6, 234, 234, 256]]],
        },
    }

    input = {
        "op": "input",
        "name": "",
        "attrs": {
            "shape": [[input_shape_a, input_shape_b, input_shape_c]],
            "dtype": [[dtype, dtype, dtype]],
        },
    }

    return [input, node]


def test_concatenate():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    for input_shape_a, input_shape_b, input_shape_c, axis, dtype in [
        ([1, 234, 234, 256], [2, 234, 234, 256], [3, 234, 234, 256], 0, "float32"),
        ([1, 1, 234, 256], [1, 2, 234, 256], [1, 3, 234, 256], 1, "float32"),
        ([1, 234, 234, 1], [1, 234, 234, 2], [1, 234, 234, 3], -1, "float32"),
        ([1, 234, 234, 256], [2, 234, 234, 256], [3, 234, 234, 256], -4, "float32"),
        ([1, 234, 234, 256], [2, 234, 234, 256], [3, 234, 234, 256], 0, "uint8"),
        ([1, 1, 234, 256], [1, 2, 234, 256], [1, 3, 234, 256], 1, "uint8"),
        ([1, 234, 234, 1], [1, 234, 234, 2], [1, 234, 234, 3], -1, "uint8"),
        ([1, 234, 234, 256], [2, 234, 234, 256], [3, 234, 234, 256], -4, "uint8"),
    ]:
        outputs = []
        inputs = {
            "a": tvm.nd.array(np.random.randn(*input_shape_a).astype(dtype)),
            "b": tvm.nd.array(np.random.randn(*input_shape_b).astype(dtype)),
            "c": tvm.nd.array(np.random.randn(*input_shape_c).astype(dtype)),
        }
        func = _get_model(
            inputs["a"].shape, inputs["b"].shape, inputs["c"].shape, axis, dtype, iter(inputs)
        )
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1, None, device, enable_acl=acl, offload_concat=True)[0])

        config = {
            "input_shape_a": input_shape_a,
            "input_shape_b": input_shape_b,
            "input_shape_c": input_shape_c,
            "axis": axis,
            "dtype": dtype,
        }
        verify(outputs, atol=1e-7, rtol=1e-7, config=config)


def test_codegen_concatenate():
    if skip_codegen_test():
        return
    shape_a = [1, 234, 234, 256]
    shape_b = [2, 234, 234, 256]
    shape_c = [3, 234, 234, 256]
    axis = 0
    inputs = {"a", "b", "c"}
    for dtype in ["float32"]:
        args = (shape_a, shape_b, shape_c, axis, dtype)
        func = _get_model(*args, iter(inputs))
        exp_codegen = _get_expected_codegen(*args)
        verify_codegen(func, exp_codegen, 1, offload_concat=True)


if __name__ == "__main__":
    test_concatenate()
    test_codegen_concatenate()

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
"""Arm Compute Library integration reshape tests."""

import numpy as np

import tvm
from tvm import relay
from tvm import testing

from .infrastructure import (
    skip_runtime_test,
    skip_codegen_test,
    build_and_run,
    verify,
    verify_codegen,
)
from .infrastructure import Device


def _get_model(input_shape, dtype, var_names):
    """Return a model and any parameters it may have."""
    a = relay.var(next(var_names), shape=input_shape, dtype=dtype)
    b = relay.var(next(var_names), shape=input_shape, dtype=dtype)
    max = relay.maximum(a, b)
    return max


def _get_expected_codegen(shape, dtype):
    node = {
        "op": "kernel",
        "name": "maximum",
        "inputs": [[0, 0, 0], [1, 0, 0]],
        "attrs": {
            "num_inputs": "2",
            "num_outputs": "1",
            "shape": [[list(shape)]],
            "dtype": [[dtype]],
        },
    }

    inputs = [
        {"op": "input", "name": "", "attrs": {"shape": [[list(shape)]], "dtype": [[dtype]]}},
        {"op": "input", "name": "", "attrs": {"shape": [[list(shape)]], "dtype": [[dtype]]}},
    ]
    inputs.append(node)
    return inputs


def test_maximum():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    for dtype, low, high, atol, rtol in [
        ("float32", -127, 128, 0.001, 0.001),
        ("float32", -1, 1, 0.001, 0.001),
    ]:
        inputs = {
            "a": tvm.nd.array(np.random.uniform(low, high, (100, 100)).astype(dtype)),
            "b": tvm.nd.array(np.random.uniform(low, high, (100, 100)).astype(dtype)),
        }
        outputs = []
        func = _get_model(inputs["a"].shape, dtype, iter(inputs))

        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1, None, device, enable_acl=acl)[0])

        verify(outputs, atol=1e-7, rtol=1e-7)


def test_codegen_maximum():
    if skip_codegen_test():
        return

    shape = (100, 100)
    inputs = {"a", "b"}
    for dtype in ["float32"]:
        args = (shape, dtype)
        func = _get_model(*args, iter(inputs))
        exp_codegen = _get_expected_codegen(*args)
        verify_codegen(func, exp_codegen, 1)


if __name__ == "__main__":
    test_maximum()
    test_codegen_maximum()

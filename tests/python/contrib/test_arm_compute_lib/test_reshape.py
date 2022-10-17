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


def _get_model(input_shape, output_shape, dtype, var_names):
    """Return a model and any parameters it may have."""
    a = relay.var(next(var_names), shape=input_shape, dtype=dtype)
    reshape = relay.reshape(a, output_shape)
    return reshape


def _get_expected_codegen(input_shape, output_shape, dtype):
    node = {
        "op": "kernel",
        "name": "reshape",
        "inputs": [[0, 0, 0]],
        "attrs": {
            "num_inputs": "1",
            "num_outputs": "1",
            "newshape": [[str(s) for s in output_shape]],
            "shape": [[list(output_shape)]],
            "dtype": [[dtype]],
            "allowzero": [["0"]],
        },
    }

    input = {
        "op": "input",
        "name": "",
        "attrs": {"shape": [[list(input_shape)]], "dtype": [[dtype]]},
    }

    return [input, node]


def test_reshape():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    for dtype, low, high, atol, rtol in [
        ("float32", -127, 128, 0.001, 0.001),
        ("uint8", 0, 255, 0, 0),
    ]:
        inputs = {"a": tvm.nd.array(np.random.uniform(low, high, (1, 1, 1, 1000)).astype(dtype))}

        for new_shape in [(1, 1000), (10, 10, 10), (10, 100, 1), (1, 1000, 1)]:
            outputs = []
            func = _get_model(inputs["a"].shape, new_shape, dtype, iter(inputs))
            for acl in [False, True]:
                outputs.append(build_and_run(func, inputs, 1, None, device, enable_acl=acl)[0])

            config = {
                "new shape": inputs["a"].shape,
                "shape": new_shape,
                "dtype": dtype,
            }
            verify(outputs, atol=1e-7, rtol=1e-7, config=config)


def test_codegen_reshape():
    if skip_codegen_test():
        return

    shape = (1, 1, 1, 1000)
    inputs = {"a"}
    for dtype in ["float32", "uint8"]:
        for new_shape in [(1, 1000), (10, 10, 10), (10, 100, 1)]:
            args = (shape, new_shape, dtype)
            func = _get_model(*args, iter(inputs))
            exp_codegen = _get_expected_codegen(*args)
            verify_codegen(func, exp_codegen, 1)


if __name__ == "__main__":
    test_reshape()
    test_codegen_reshape()

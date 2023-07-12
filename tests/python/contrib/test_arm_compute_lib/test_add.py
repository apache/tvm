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
import pytest

import tvm
import tvm.testing
from tvm import relay

from test_arm_compute_lib.infrastructure import (
    skip_runtime_test,
    skip_codegen_test,
    build_and_run,
    verify,
    verify_codegen,
)
from test_arm_compute_lib.infrastructure import Device

_qnn_params = {
    "lhs_scale": relay.const(0.0156863, "float32"),
    "lhs_zero_point": relay.const(127, "int32"),
    "rhs_scale": relay.const(0.0117647, "float32"),
    "rhs_zero_point": relay.const(85, "int32"),
    "output_scale": relay.const(0.0235294, "float32"),
    "output_zero_point": relay.const(128, "int32"),
}


def _get_model(shape, dtype, var_names, op, op_params):
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    b = relay.var(next(var_names), shape=shape, dtype=dtype)
    return op(a, b, **op_params)


def _get_expected_codegen(shape, dtype, op_name, qnn_params):
    input_a = {"op": "input", "name": "", "attrs": {"shape": [[list(shape)]], "dtype": [[dtype]]}}
    input_b = {"op": "input", "name": "", "attrs": {"shape": [[list(shape)]], "dtype": [[dtype]]}}
    input_qnn = [
        {
            "op": "const",
            "name": "",
            "attrs": {
                "shape": [[list(qnn_params[_].data.shape)]],
                "dtype": [[qnn_params[_].data.dtype]],
            },
        }
        for _ in qnn_params
    ]
    inputs = [input_a, input_b, *input_qnn]
    node = {
        "op": "kernel",
        "name": op_name,
        "inputs": [[_, 0, 0] for _ in range(len(inputs))],
        "attrs": {
            "num_inputs": str(len(inputs)),
            "num_outputs": "1",
            "shape": [[list(shape)]],
            "dtype": [[dtype]],
        },
    }

    if qnn_params:
        node["attrs"]["lhs_axis"] = [["-1"]]
        node["attrs"]["rhs_axis"] = [["-1"]]

    return [*inputs, node]


def test_runtime_add():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    for dtype, low, high, atol, rtol, op, op_params in [
        ("float32", -127, 128, 1e-7, 1e-7, relay.add, {}),
        ("uint8", 0, 255, 1.0, 0.0, relay.qnn.op.add, _qnn_params),
        ("int8", -127, 128, 1.0, 0.0, relay.qnn.op.add, _qnn_params),
    ]:
        shape = (2, 2)
        for inputs in [
            {
                "a": tvm.nd.array(np.random.uniform(low, high, shape).astype(dtype)),
                "b": tvm.nd.array(np.random.uniform(low, high, shape).astype(dtype)),
            }
        ]:
            outputs = []
            func = _get_model(shape, dtype, iter(inputs), op, op_params)
            for acl in [True, False]:
                outputs.append(build_and_run(func, inputs, 1, None, device, enable_acl=acl)[0])

            config = {
                "shape": shape,
                "dtype": dtype,
                "inputs": inputs,
                "operation": op,
                "op_params": op_params,
            }

            verify(outputs, atol=atol, rtol=rtol, config=config, verify_saturation=False)


def test_codegen_add():
    if skip_codegen_test():
        return

    inputs = {"a", "b"}
    for dtype, op_name, op, qnn_params in [
        ("float32", "add", relay.add, {}),
        ("uint8", "qnn.add", relay.qnn.op.add, _qnn_params),
        ("int8", "qnn.add", relay.qnn.op.add, _qnn_params),
    ]:
        for shape in [(1, 1), (2, 2, 2), (3, 3, 3, 3)]:
            func = _get_model(shape, dtype, iter(inputs), op, qnn_params)
            exp_codegen = _get_expected_codegen(shape, dtype, op_name, qnn_params)
            verify_codegen(func, exp_codegen, 1)


@pytest.mark.parametrize(
    "param, param_type",
    [
        ("lhs_scale", "float32"),
        ("lhs_zero_point", "int32"),
        ("rhs_scale", "float32"),
        ("rhs_zero_point", "int32"),
    ],
)
def test_codegen_add_per_channel_quantization(param, param_type):
    if skip_codegen_test():
        return

    qnn_params = _qnn_params
    qnn_params[param] = relay.const([1, 2], param_type)

    dtype = "int8"
    op_name = "qnn.add"
    op = relay.qnn.op.add
    inputs = {"a", "b"}

    for shape in [(1, 3, 3, 2)]:
        func = _get_model(shape, dtype, iter(inputs), op, qnn_params)
        exp_codegen = _get_expected_codegen(shape, dtype, op_name, qnn_params)
        verify_codegen(func, exp_codegen, num_acl_modules=0, tvm_ops=1)


if __name__ == "__main__":
    test_runtime_add()
    test_codegen_add()
    test_codegen_add_per_channel_quantization()

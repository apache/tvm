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
"""BNNS integration dense tests."""

import numpy as np
import math
import pytest
import tvm
from tvm import relay
from .infrastructure import (
    Device,
    skip_runtime_test,
    skip_codegen_test,
    build_and_run,
    verify,
    verify_codegen,
    generate_trials,
)


def _get_model(shape, weight_shape, units, dtype, var_names, has_bias=False, has_gelu=False):
    """Return a model and any parameters it may have"""
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.dense(a, weights, units=units, out_dtype=dtype)
    params = {"w": w}
    if has_bias:
        b = tvm.nd.array(np.random.randint(-128, 127, weight_shape[0]).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.op.add(out, biasc)
        params["b"] = b
    if has_gelu:
        const1 = relay.const(0.044715)
        const2 = relay.const(math.sqrt(2 / math.pi))
        bias = out
        out = relay.op.power(bias, relay.const(3.0, "float32"))
        out = relay.op.multiply(out, const1)
        out = relay.op.add(out, bias)
        out = relay.op.multiply(out, const2)
        out = relay.op.tanh(out)
        out = relay.op.add(out, relay.const(1, "float32"))
        out = relay.op.multiply(out, relay.const(0.5))
        out = relay.op.multiply(out, bias)
    return out, params


def _get_expected_codegen(shape, weight_shape, units, dtype, has_bias=False, has_gelu=False):
    output_shape = (shape[0], units)
    name = "nn.dense"
    if has_bias is True:
        name = "bnns.dense_bias"
    if has_bias is True and has_gelu is True:
        name = "bnns.dense_bias_gelu"

    node = {
        "op": "kernel",
        "name": name,
        "inputs": [],
        "attrs": {
            "num_outputs": "1",
            "out_dtype": [["float32"]],
            "shape": [[list(output_shape)]],
            "dtype": [[dtype]],
            "units": [[str(units)]],
        },
    }

    inputs = [
        {"op": "input", "name": "", "attrs": {"shape": [[list(shape)]], "dtype": [[str(dtype)]]}},
        {
            "op": "const",
            "name": "",
            "attrs": {"shape": [[list(weight_shape)]], "dtype": [[str(dtype)]]},
        },
    ]

    if has_bias:
        inputs.append(
            {
                "op": "const",
                "name": "",
                "attrs": {"shape": [[[weight_shape[0]]]], "dtype": [["float32"]]},
            }
        )

    input_idx = 0
    for _ in range(len(inputs)):
        node["inputs"].append([input_idx, 0, 0])
        input_idx += 1
    node["attrs"]["num_inputs"] = str(len(inputs))
    inputs.append(node)
    return inputs


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because BNNS codegen is not available")
def test_dense():
    device = Device()
    np.random.seed(0)

    dtype = ["float32"]
    shape = [
        ((1, 128), (16, 128), 16),
        ((32, 32), (32, 32), 32),
        ((1, 64), (1, 64), 1),
        ((11, 2), (2, 2), 2),
        ((2, 2), (1, 2), 1),
    ]
    composite = [False, True]
    trials = generate_trials([dtype, shape, composite, composite], 3)

    for dtype, (shape, weight_shape, units), with_bias, with_gelu in trials:
        outputs = []
        inputs = {"a": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype))}
        func, params = _get_model(
            shape,
            weight_shape,
            units,
            dtype,
            var_names=iter(inputs),
            has_bias=with_bias,
            has_gelu=with_gelu,
        )
        for bnns in [False, True]:
            outputs.append(
                build_and_run(
                    func,
                    inputs,
                    1,
                    params,
                    device,
                    enable_bnns=bnns,
                )[0]
            )

        config = {
            "shape": shape,
            "weight_shape": weight_shape,
            "units": units,
            "dtype": dtype,
            "with_bias": with_bias,
            "with_gelu": with_gelu,
        }
        verify(outputs, atol=0.001, rtol=0.01, config=config)


@pytest.mark.skipif(skip_codegen_test(), reason="Skip because BNNS codegen is not available")
def test_codegen_dense():
    np.random.seed(0)

    dtype = ["float32"]
    shape = [
        ((1, 128), (16, 128), 16),
        ((32, 32), (32, 32), 32),
        ((1, 64), (1, 64), 1),
        ((11, 2), (2, 2), 2),
        ((2, 2), (1, 2), 1),
    ]
    composite = [False, True]
    trials = generate_trials([dtype, shape, composite, composite], 3)

    for dtype, (shape, weight_shape, units), with_bias, with_gelu in trials:
        inputs = {"a"}

        args = (shape, weight_shape, units, dtype)

        func, params = _get_model(
            *args, var_names=iter(inputs), has_bias=with_bias, has_gelu=with_gelu
        )
        exp_codegen = _get_expected_codegen(*args, has_bias=with_bias, has_gelu=with_gelu)
        verify_codegen(func, exp_codegen, 1)


if __name__ == "__main__":
    test_dense()
    test_codegen_dense()

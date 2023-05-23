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
"""Arm Compute Library integration dense tests."""
import numpy as np
import pytest

import tvm
from tvm import relay
from tvm import testing
from test_arm_compute_lib.infrastructure import (
    Device,
    skip_runtime_test,
    skip_codegen_test,
    build_and_run,
    verify,
    verify_codegen,
)


def _get_model(shape, weight_shape, units, dtype, var_names, has_bias=False):
    """Return a model and any parameters it may have"""
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.dense(a, weights, units=units, out_dtype=dtype)
    params = {"w": w}
    if has_bias:
        b = tvm.nd.array(np.random.randint(-128, 127, weight_shape[0]).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.nn.bias_add(out, biasc)
        params["b"] = b
    return out, params


def _get_qnn_params(input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w):
    """Get output qnn parameters given input and kernel parameters."""
    input_max = input_sc * (255 - input_zp)
    input_min = -input_sc * input_zp
    kernel_max = kernel_sc * (255 - kernel_zp)
    kernel_min = -kernel_sc * kernel_zp
    output_limits = [
        kernel_max * kernel_h * kernel_w * input_max,
        kernel_min * kernel_h * kernel_w * input_max,
        kernel_min * kernel_h * kernel_w * input_min,
        kernel_max * kernel_h * kernel_w * input_min,
    ]
    output_max = max(output_limits)
    output_min = min(output_limits)
    output_sc = (output_max - output_min) / 255
    output_zp = -int(output_min / output_sc)
    return output_zp, output_sc


def _get_qnn_model(
    shape,
    weight_shape,
    units,
    dtype,
    input_zp,
    input_sc,
    kernel_zp,
    kernel_sc,
    output_zp,
    output_sc,
    var_names,
    has_bias=False,
):
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.qnn.op.dense(
        a,
        weights,
        units=units,
        input_zero_point=relay.const(input_zp, "int32"),
        kernel_zero_point=relay.const(kernel_zp, "int32"),
        input_scale=relay.const(input_sc, "float32"),
        kernel_scale=relay.const(kernel_sc, "float32"),
        out_dtype="int32",
    )
    params = {"w": w}
    if has_bias:
        b = tvm.nd.array(np.random.randint(0, 255, weight_shape[0]).astype("int32"))
        biasc = relay.const(b, "int32")
        out = relay.nn.bias_add(out, biasc)
        params["b"] = b
    out = relay.qnn.op.requantize(
        out,
        relay.const(input_sc * kernel_sc, "float32"),  # input scale
        relay.const(0, "int32"),  # input zero point
        relay.const(output_sc, "float32"),  # output scale
        relay.const(output_zp, "int32"),  # output zero point
        out_dtype=dtype,
    )
    return out, params


def _get_expected_codegen(shape, weight_shape, units, dtype, has_bias=False):
    output_shape = (shape[0], units)
    qnn_dtypes = ("uint8", "int8")
    out_dtype = "int32" if dtype in qnn_dtypes else "float32"

    node = {
        "op": "kernel",
        "name": "nn.dense",
        "inputs": [],
        "attrs": {
            "num_outputs": "1",
            "out_dtype": [[out_dtype]],
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

    # qnn.dense params, input and kernel
    if dtype in qnn_dtypes:
        node["name"] = "qnn.dense"
        for param_dtype in ["int32", "float32"]:
            for _ in range(2):
                inputs.append(
                    {
                        "op": "const",
                        "name": "",
                        "attrs": {"shape": [[[]]], "dtype": [[param_dtype]]},
                    }
                )

    if has_bias:
        bias_dtype = "int32" if dtype in qnn_dtypes else "float32"
        bias_shape = [1, weight_shape[0]] if weight_shape[0] != 1 else [weight_shape[0]]
        inputs.append(
            {
                "op": "const",
                "name": "",
                "attrs": {"shape": [[bias_shape]], "dtype": [[bias_dtype]]},
            }
        )

    # qnn.dense params, output
    if dtype in qnn_dtypes:
        for param_dtype in ["float32", "int32"]:
            inputs.append(
                {"op": "const", "name": "", "attrs": {"shape": [[[]]], "dtype": [[param_dtype]]}}
            )

    input_idx = 0
    for _ in range(len(inputs)):
        node["inputs"].append([input_idx, 0, 0])
        input_idx += 1
    node["attrs"]["num_inputs"] = str(len(inputs))
    inputs.append(node)
    return inputs


def test_dense():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)
    dtype = "float32"
    trials = [
        [(1, 128), (16, 128), 16, True],
        [(1, 128), (16, 128), 16, False],
        [(32, 32), (32, 32), 32, True],
        [(32, 32), (32, 32), 32, False],
        [(1, 64), (1, 64), 1, True],
        [(1, 64), (1, 64), 1, False],
        [(11, 2), (2, 2), 2, True],
        [(11, 2), (2, 2), 2, False],
    ]
    for shape, weight_shape, units, composite in trials:
        outputs = []
        inputs = {"a": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype))}
        func, params = _get_model(
            shape, weight_shape, units, dtype, var_names=iter(inputs), has_bias=composite
        )
        for acl in [False, True]:
            outputs.append(
                build_and_run(
                    func,
                    inputs,
                    1,
                    params,
                    device,
                    enable_acl=acl,
                )[0]
            )
        config = {
            "shape": shape,
            "weight_shape": weight_shape,
            "units": units,
            "dtype": dtype,
            "composite operators (bias)": composite,
        }
        verify(outputs, atol=0.001, rtol=0.01, config=config)


def test_codegen_dense():
    if skip_codegen_test():
        return

    np.random.seed(0)
    dtype = "float32"
    trials = [
        [(1, 128), (16, 128), 16, True],
        [(1, 128), (16, 128), 16, False],
        [(32, 32), (32, 32), 32, True],
        [(32, 32), (32, 32), 32, False],
        [(1, 64), (1, 64), 1, True],
        [(1, 64), (1, 64), 1, False],
        [(11, 2), (2, 2), 2, True],
        [(11, 2), (2, 2), 2, False],
    ]
    for shape, weight_shape, units, composite in trials:
        inputs = {"a"}

        args = (shape, weight_shape, units, dtype)

        func, params = _get_model(*args, var_names=iter(inputs), has_bias=composite)
        exp_codegen = _get_expected_codegen(*args, has_bias=composite)
        verify_codegen(func, exp_codegen)


@pytest.mark.parametrize(
    "dtype,min_range,max_range",
    [
        ("uint8", 0, 255),
        ("int8", -127, 128),
    ],
)
def test_qnn_dense(dtype, min_range, max_range):
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    trials = [
        [(1, 2), (2, 2), 2, True],
        [(1, 2), (2, 2), 2, False],
        [(4, 4), (4, 4), 4, True],
        [(4, 4), (4, 4), 4, False],
        [(16, 16), (4, 16), 4, True],
        [(16, 16), (4, 16), 4, False],
        [(1, 128), (16, 128), 16, True],
        [(1, 128), (16, 128), 16, False],
        [(32, 32), (32, 32), 32, True],
        [(32, 32), (32, 32), 32, False],
        [(1, 64), (1, 64), 1, True],
        [(1, 64), (1, 64), 1, False],
    ]
    for shape, weight_shape, units, composite in trials:
        outputs = []
        inputs = {"a": tvm.nd.array(np.random.uniform(min_range, max_range, shape).astype(dtype))}
        input_zp = 100
        input_sc = 0.5
        kernel_zp = 50
        kernel_sc = 0.03
        output_zp, output_sc = _get_qnn_params(
            input_zp, input_sc, kernel_zp, kernel_sc, weight_shape[0], weight_shape[1]
        )

        func, params = _get_qnn_model(
            shape,
            weight_shape,
            units,
            dtype,
            input_zp,
            input_sc,
            kernel_zp,
            kernel_sc,
            output_zp,
            output_sc,
            var_names=iter(inputs),
            has_bias=composite,
        )

        for acl in [False, True]:
            outputs.append(
                build_and_run(
                    func,
                    inputs,
                    1,
                    params,
                    device,
                    enable_acl=acl,
                )[0]
            )

        config = {
            "shape": shape,
            "weight_shape": weight_shape,
            "units": units,
            "dtype": dtype,
            "composite operators (bias)": composite,
            "input scale": input_sc,
            "input zero point": input_zp,
            "kernel scale": kernel_sc,
            "kernel zero point": kernel_zp,
            "output scale": output_sc,
            "output zero point": output_zp,
        }
        verify(outputs, atol=1, rtol=0, config=config, verify_saturation=True)


@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_codegen_qnn_dense(dtype):
    if skip_codegen_test():
        return

    np.random.seed(0)

    trials = [
        [(1, 2), (2, 2), 2, True],
        [(1, 2), (2, 2), 2, False],
        [(4, 4), (4, 4), 4, True],
        [(4, 4), (4, 4), 4, False],
        [(16, 16), (4, 16), 4, True],
        [(16, 16), (4, 16), 4, False],
        [(1, 128), (16, 128), 16, True],
        [(1, 128), (16, 128), 16, False],
        [(32, 32), (32, 32), 32, True],
        [(32, 32), (32, 32), 32, False],
        [(1, 64), (1, 64), 1, True],
        [(1, 64), (1, 64), 1, False],
    ]
    for shape, weight_shape, units, composite in trials:
        inputs = {"a"}
        args = (shape, weight_shape, units, dtype)

        input_zp = 100
        input_sc = 0.5
        kernel_zp = 25
        kernel_sc = 0.03
        output_zp, output_sc = _get_qnn_params(
            input_zp, input_sc, kernel_zp, kernel_sc, weight_shape[0], weight_shape[1]
        )

        func, params = _get_qnn_model(
            *args,
            var_names=iter(inputs),
            input_zp=input_zp,
            input_sc=input_sc,
            kernel_zp=kernel_zp,
            kernel_sc=kernel_sc,
            output_zp=output_zp,
            output_sc=output_sc,
            has_bias=composite,
        )
        exp_codegen = _get_expected_codegen(*args, has_bias=composite)
        verify_codegen(func, exp_codegen)


@pytest.mark.parametrize(
    "param",
    ["kernel_sc", "kernel_zp"],
)
def test_codegen_qnn_dense_per_channel_quantization(param):
    if skip_codegen_test():
        return

    np.random.seed(0)
    dtype = "int8"
    shape = (1, 2)
    weight_shape = (2, 2)
    units = 2
    composite = True
    inputs = {"a"}
    args = (shape, weight_shape, units, dtype)

    qnn_params = {
        "input_zp": 1,
        "input_sc": 1,
        "kernel_zp": 1,
        "kernel_sc": 1,
        "output_zp": 1,
        "output_sc": 1,
    }
    qnn_params[param] = [1, 1]

    func, _ = _get_qnn_model(
        *args,
        var_names=iter(inputs),
        input_zp=qnn_params["input_zp"],
        input_sc=qnn_params["input_sc"],
        kernel_zp=qnn_params["kernel_zp"],
        kernel_sc=qnn_params["kernel_sc"],
        output_zp=qnn_params["output_zp"],
        output_sc=qnn_params["output_sc"],
        has_bias=composite,
    )
    exp_codegen = _get_expected_codegen(*args, has_bias=composite)
    verify_codegen(func, exp_codegen, num_acl_modules=0, tvm_ops=3)


if __name__ == "__main__":
    test_dense()
    test_qnn_dense()
    test_codegen_dense()
    test_codegen_qnn_dense()
    test_codegen_qnn_dense_per_channel_quantization()

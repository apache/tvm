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
"""BNNS integration pooling tests."""

import numpy as np
import pytest
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


def _calculate_output_shape(shape, sizes, padding, strides):
    """Calculate pooling output shape."""
    output_height = ((shape[2] - sizes[0] + padding[0] + padding[2]) / strides[0]) + 1
    output_width = ((shape[3] - sizes[1] + padding[1] + padding[3]) / strides[1]) + 1
    return 1, shape[1], int(output_height), int(output_width)


def _get_pooling_model(
    shape, dtype, typef, sizes, strides, padding, ceil_mode, count_include_pad, var_names
):
    """Return a model and any parameters it may have."""
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    out = relay.var(next(var_names), shape=shape, dtype=dtype)

    if typef == "nn.max_pool2d":
        out = relay.nn.max_pool2d(
            out,
            pool_size=sizes,
            strides=strides,
            padding=padding,
            ceil_mode=ceil_mode,
        )
    elif typef == "nn.avg_pool2d":
        out = relay.nn.avg_pool2d(
            out,
            pool_size=sizes,
            strides=strides,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
    else:
        raise ValueError("Function not supported")

    return out


def _get_global_pooling_model(shape, dtype, typef, var_names):
    """Return a model and any parameters it may have."""
    out = relay.var(next(var_names), shape=shape, dtype=dtype)

    if typef == "nn.global_max_pool2d":
        out = relay.nn.global_max_pool2d(out)
    elif typef == "nn.global_avg_pool2d":
        out = relay.nn.global_avg_pool2d(out)
    else:
        raise ValueError("Function not supported")

    return out


def _get_expected_pooling_codegen(
    shape, dtype, typef, sizes, strides, padding, ceil_mode, count_include_pad
):
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    output_shape = _calculate_output_shape(shape, sizes, padding, strides)

    node = {
        "op": "kernel",
        "name": typef,
        "inputs": [[0, 0, 0]],
        "attrs": {
            "num_inputs": "1",
            "num_outputs": "1",
            "layout": [["NCHW"]],
            "shape": [[list(output_shape)]],
            "dtype": [[dtype]],
            "padding": [[str(p) for p in padding]],
            "strides": [[str(s) for s in strides]],
            "pool_size": [[str(s) for s in sizes]],
            "ceil_mode": [[str(1 if ceil_mode else 0)]],
        },
    }

    if typef == "nn.avg_pool2d" or typef == "nn.l2_pool2d":
        node["attrs"]["count_include_pad"] = [["1" if count_include_pad else "0"]]

    input = {"op": "input", "name": "", "attrs": {"shape": [[list(shape)]], "dtype": [[dtype]]}}
    return [input, node]


def _get_expected_global_pooling_codegen(shape, dtype, typef):
    node = {
        "op": "kernel",
        "name": typef,
        "inputs": [[0, 0, 0]],
        "attrs": {
            "num_inputs": "1",
            "num_outputs": "1",
            "layout": [["NCHW"]],
            "shape": [[[1, shape[1], 1, 1]]],
            "dtype": [[dtype]],
        },
    }

    input = {"op": "input", "name": "", "attrs": {"shape": [[list(shape)]], "dtype": [[dtype]]}}
    return [input, node]


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because BNNS codegen is not available")
def test_pooling():
    device = Device()
    np.random.seed(0)

    dtype = "float32"
    trials = [
        ["nn.max_pool2d", (3, 3), (2, 2), (0, 0), False, False, (27, 27, 512)],
        ["nn.max_pool2d", (2, 2), (2, 2), (0, 0), False, True, (16, 16, 16)],
        ["nn.max_pool2d", (3, 3), (2, 2), (1, 1), True, True, (15, 15, 16)],
        ["nn.max_pool2d", (2, 2), (2, 2), (0, 1), False, False, (16, 16, 16)],
        ["nn.avg_pool2d", (2, 2), (2, 2), (1, 1), False, False, (16, 16, 16)],
        ["nn.avg_pool2d", (2, 2), (2, 2), (0, 0), False, True, (16, 16, 16)],
        ["nn.avg_pool2d", (3, 3), (2, 2), (0, 1), True, False, (15, 15, 16)],
    ]

    for (
        typef,
        size,
        stride,
        pad,
        ceil_mode,
        count_include_pad,
        input_shape,
    ) in trials:
        shape = (1, *input_shape)
        outputs = []
        inputs = {
            "a": tvm.nd.array(np.random.uniform(-127, 128, shape).astype(dtype)),
        }

        func = _get_pooling_model(
            shape, dtype, typef, size, stride, pad, ceil_mode, count_include_pad, iter(inputs)
        )

        config = {
            "size": size,
            "stride": stride,
            "shape": shape,
            "pooling type": typef,
            "dtype": dtype,
            "padding": pad,
            "ceil_mode": ceil_mode,
            "count_include_pad": count_include_pad,
            "inputs": inputs,
        }

        params = None
        for enable_bnns in [False, True]:
            outputs.append(
                build_and_run(
                    func, inputs, 1, params, device, enable_bnns=enable_bnns, config=config
                )[0]
            )

        verify(outputs, atol=0.001, rtol=0.001, config=config)


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because BNNS codegen is not available")
def test_global_pooling():
    device = Device()
    np.random.seed(0)

    dtype = "float32"

    trials = [
        ["nn.global_max_pool2d", (8, 8, 16)],
        ["nn.global_max_pool2d", (9, 9, 16)],
        ["nn.global_max_pool2d", (8, 8, 16)],
        ["nn.global_avg_pool2d", (8, 8, 16)],
        ["nn.global_avg_pool2d", (8, 8, 16)],
        ["nn.global_avg_pool2d", (9, 9, 16)],
    ]

    for typef, input_shape in trials:
        shape = (1, *input_shape)
        outputs = []
        inputs = {
            "a": tvm.nd.array(np.random.uniform(-127, 128, shape).astype(dtype)),
        }

        func = _get_global_pooling_model(shape, dtype, typef, iter(inputs))
        config = {
            "shape": shape,
            "pooling type": typef,
            "dtype": dtype,
        }

        for enable_bnns in [False, True]:
            outputs.append(
                build_and_run(
                    func, inputs, 1, None, device, enable_bnns=enable_bnns, config=config
                )[0]
            )

        verify(outputs, atol=0.001, rtol=0.001, config=config)


@pytest.mark.skipif(skip_codegen_test(), reason="Skip because BNNS codegen is not available")
def test_codegen_pooling():
    dtype = "float32"

    trials = [
        ["nn.max_pool2d", (2, 2), (2, 2), (0, 0), False, True, (16, 16, 16)],
        ["nn.max_pool2d", (3, 3), (2, 2), (1, 1), True, True, (15, 15, 16)],
        ["nn.max_pool2d", (2, 2), (2, 2), (0, 1), False, False, (16, 16, 16)],
        ["nn.avg_pool2d", (2, 2), (2, 2), (1, 1), False, False, (16, 16, 16)],
        ["nn.avg_pool2d", (2, 2), (2, 2), (0, 0), False, True, (16, 16, 16)],
        ["nn.avg_pool2d", (3, 3), (2, 2), (0, 1), True, False, (15, 15, 16)],
    ]

    for (
        typef,
        size,
        stride,
        pad,
        ceil_mode,
        count_include_pad,
        input_shape,
    ) in trials:
        shape = (1, *input_shape)
        inputs = {"a"}
        args = (shape, dtype, typef, size, stride, pad, False, False)
        func = _get_pooling_model(*args, iter(inputs))
        exp_codegen = _get_expected_pooling_codegen(*args)
        verify_codegen(func, exp_codegen, 1)


@pytest.mark.skipif(skip_codegen_test(), reason="Skip because BNNS codegen is not available")
def test_codegen_global_pooling():
    dtype = "float32"

    trials = [
        ["nn.global_max_pool2d", (8, 8, 16)],
        ["nn.global_max_pool2d", (9, 9, 16)],
        ["nn.global_max_pool2d", (8, 8, 16)],
        ["nn.global_avg_pool2d", (8, 8, 16)],
        ["nn.global_avg_pool2d", (8, 8, 16)],
        ["nn.global_avg_pool2d", (9, 9, 16)],
    ]

    for typef, input_shape in trials:
        shape = (1, *input_shape)
        inputs = {"a"}
        args = (shape, dtype, typef)
        func = _get_global_pooling_model(*args, iter(inputs))
        exp_codegen = _get_expected_global_pooling_codegen(*args)
        verify_codegen(func, exp_codegen, 1)


if __name__ == "__main__":
    test_pooling()
    test_global_pooling()
    test_codegen_pooling()
    test_codegen_global_pooling()

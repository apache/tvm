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
import pytest

import tvm
from tvm import relay, testing

from test_arm_compute_lib.infrastructure import (
    Device,
    build_and_run,
    skip_codegen_test,
    skip_runtime_test,
    verify,
    verify_codegen,
)


def _calculate_output_shape(shape, sizes, padding, strides, dilation):
    """Calculate pooling output shape."""
    height_receptive_field = (sizes[0] - 1) * dilation[0] + 1
    width_receptive_field = (sizes[1] - 1) * dilation[1] + 1
    output_height = ((shape[1] - height_receptive_field + padding[0] + padding[2]) / strides[0]) + 1
    output_width = ((shape[2] - width_receptive_field + padding[1] + padding[3]) / strides[1]) + 1
    return 1, int(output_height), int(output_width), shape[3]


def _get_pooling_model(
    shape, dtype, typef, sizes, strides, dilation, padding, ceil_mode, count_include_pad, var_names
):
    """Return a model and any parameters it may have."""
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    out = relay.var(next(var_names), shape=shape, dtype=dtype)
    qnn_dtypes = ("uint8", "int8")

    if typef == "nn.max_pool2d":
        out = relay.nn.max_pool2d(
            out,
            pool_size=sizes,
            strides=strides,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
            layout="NHWC",
        )
    elif typef == "nn.avg_pool2d":
        if dtype in qnn_dtypes:
            out = relay.cast(out, "int32")
        out = relay.nn.avg_pool2d(
            out,
            pool_size=sizes,
            strides=strides,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            layout="NHWC",
        )
        if dtype in qnn_dtypes:
            out = relay.cast(out, dtype)
    elif typef == "nn.l2_pool2d":
        out = relay.power(out, relay.const(2.0))
        out = relay.nn.avg_pool2d(
            out,
            pool_size=sizes,
            strides=strides,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            layout="NHWC",
        )
        out = relay.sqrt(out)
    else:
        raise ValueError("Function not supported")

    return out


def _get_global_pooling_model(shape, dtype, typef, var_names):
    """Return a model and any parameters it may have."""
    out = relay.var(next(var_names), shape=shape, dtype=dtype)
    qnn_dtypes = ("uint8", "int8")

    if typef == "nn.global_max_pool2d":
        out = relay.nn.global_max_pool2d(out, layout="NHWC")
    elif typef == "nn.global_avg_pool2d":
        if dtype in qnn_dtypes:
            out = relay.cast(out, "int32")
        out = relay.nn.global_avg_pool2d(out, layout="NHWC")
        if dtype in qnn_dtypes:
            out = relay.cast(out, dtype)
    else:
        raise ValueError("Function not supported")

    return out


def _get_expected_pooling_codegen(
    shape, dtype, typef, sizes, strides, dilation, padding, ceil_mode, count_include_pad
):
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    output_shape = _calculate_output_shape(shape, sizes, padding, strides, dilation)

    node = {
        "op": "kernel",
        "name": typef,
        "inputs": [[0, 0, 0]],
        "attrs": {
            "num_inputs": "1",
            "num_outputs": "1",
            "layout": [["NHWC"]],
            "out_layout": [[""]],
            "shape": [[list(output_shape)]],
            "dtype": [[dtype]],
            "padding": [[str(p) for p in padding]],
            "strides": [[str(s) for s in strides]],
            "dilation": [[str(d) for d in dilation]],
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
            "layout": [["NHWC"]],
            "out_layout": [[""]],
            "shape": [[[1, 1, 1, shape[3]]]],
            "dtype": [[dtype]],
        },
    }

    input = {"op": "input", "name": "", "attrs": {"shape": [[list(shape)]], "dtype": [[dtype]]}}
    return [input, node]


def _get_low_high_atol_rtol(dtype):
    if dtype == "float32":
        low, high, atol, rtol = (-127, 128, 0.001, 0.001)
    elif dtype == "uint8":
        low, high, atol, rtol = (0, 255, 1, 0)
    elif dtype == "int8":
        low, high, atol, rtol = (-127, 128, 1, 0)
    else:
        pytest.fail(f"dtype not expected: {dtype}")

    return low, high, atol, rtol


# fmt: off
@pytest.mark.parametrize(
     "typef,dtype,size,stride,dilation,pad,ceil_mode,count_include_pad,input_shape,expected_ops",
     [
        ("nn.max_pool2d", "float32",  (3, 3), (2, 2), (1, 1), (0, 0), False, False, (27, 27, 512), (0, 1),),
        ("nn.max_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (0, 0), False, True,  (16, 16, 16),  (0, 1),),
        ("nn.max_pool2d", "float32",  (3, 3), (2, 2), (1, 1), (1, 1), True,  True,  (15, 15, 16),  (0, 1),),
        ("nn.max_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16),  (0, 1),),
        ("nn.max_pool2d", "uint8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16),  (0, 1),),
        ("nn.max_pool2d", "uint8", (2, 2), (2, 2), (1, 1), (1, 1), True,  True,  (15, 15, 16),  (0, 1),),
        ("nn.max_pool2d", "uint8", (2, 2), (2, 2), (3, 2), (1, 1), True,  True,  (15, 15, 16),  (1, 0),),
        ("nn.max_pool2d", "int8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16),  (0, 1),),
        ("nn.max_pool2d", "int8", (2, 2), (2, 2), (1, 1), (1, 1), True,  True,  (15, 15, 16),  (0, 1),),
        ("nn.max_pool2d", "int8", (2, 2), (2, 2), (3, 2), (1, 1), True,  True,  (15, 15, 16),  (1, 0),),
        ("nn.avg_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (1, 1), False, False, (16, 16, 16),  (0, 1),),
        ("nn.avg_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (0, 0), False, True,  (16, 16, 16),  (0, 1),),
        ("nn.avg_pool2d", "float32",  (3, 3), (2, 2), (3, 2), (0, 1), True,  False, (15, 15, 16),  (1, 0),),
        # 20.05: "exclude_padding equal false is not supported for AVG Pooling with padding on quantized types"
        # ["nn.avg_pool2d", uint8_dtype, (2, 2), (2, 2), (1, 1), False, True, (16, 16, 16)],
        ("nn.avg_pool2d", "uint8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16),  (0, 1),),
        ("nn.avg_pool2d", "int8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16),  (0, 1),),
        ("nn.l2_pool2d",  "float32",  (2, 2), (2, 2), (1, 1), (0, 1), True,  False, (16, 16, 16),  (0, 1),),
        ("nn.l2_pool2d",  "float32",  (3, 3), (2, 2), (1, 1), (0, 0), False, False, (16, 16, 16),  (0, 1),),
        ("nn.l2_pool2d",  "float32",  (2, 2), (2, 2), (1, 1), (1, 1), False, True,  (15, 15, 16),  (0, 1),),

     ],
)
# fmt: on
def test_pooling(
    typef,
    dtype,
    size,
    stride,
    dilation,
    pad,
    ceil_mode,
    count_include_pad,
    input_shape,
    expected_ops,
):
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    low, high, atol, rtol = _get_low_high_atol_rtol(dtype)
    tvm_ops, acl_partitions = expected_ops

    shape = (1, *input_shape)
    outputs = []
    inputs = {
        "a": tvm.nd.array(np.random.uniform(low, high, shape).astype(dtype)),
    }

    func = _get_pooling_model(
        shape,
        dtype,
        typef,
        size,
        stride,
        dilation,
        pad,
        ceil_mode,
        count_include_pad,
        iter(inputs),
    )

    config = {
        "size": size,
        "stride": stride,
        "shape": shape,
        "pooling type": typef,
        "dtype": dtype,
        "padding": pad,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "inputs": inputs,
    }
    verify_saturation = True if dtype == "uint8" else False
    for acl in [False, True]:
        outputs.append(
            build_and_run(
                func,
                inputs,
                1,
                None,
                device,
                enable_acl=acl,
                tvm_ops=tvm_ops,
                acl_partitions=acl_partitions,
                config=config,
            )[0]
        )

    verify(outputs, atol=atol, rtol=rtol, config=config, verify_saturation=verify_saturation)


@pytest.mark.parametrize(
    "typef,dtype,input_shape",
    [
        ["nn.global_max_pool2d", "float32", (8, 8, 16)],
        ["nn.global_max_pool2d", "float32", (9, 9, 16)],
        ["nn.global_max_pool2d", "uint8", (8, 8, 16)],
        ["nn.global_max_pool2d", "uint8", (9, 9, 16)],
        ["nn.global_max_pool2d", "int8", (8, 8, 16)],
        ["nn.global_max_pool2d", "int8", (9, 9, 16)],
        ["nn.global_avg_pool2d", "float32", (8, 8, 16)],
        ["nn.global_avg_pool2d", "float32", (9, 9, 16)],
        ["nn.global_avg_pool2d", "uint8", (8, 8, 16)],
        ["nn.global_avg_pool2d", "uint8", (9, 9, 16)],
        ["nn.global_avg_pool2d", "int8", (8, 8, 16)],
        ["nn.global_avg_pool2d", "int8", (9, 9, 16)],
    ],
)
def test_global_pooling(typef, dtype, input_shape):
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    low, high, rtol, atol = _get_low_high_atol_rtol(dtype)

    shape = (1, *input_shape)
    outputs = []
    inputs = {
        "a": tvm.nd.array(np.random.uniform(low, high, shape).astype(dtype)),
    }

    func = _get_global_pooling_model(shape, dtype, typef, iter(inputs))
    config = {
        "shape": shape,
        "pooling type": typef,
        "dtype": dtype,
    }
    verify_saturation = True if dtype in ("uint8", "int8") else False

    for acl in [False, True]:
        outputs.append(
            build_and_run(func, inputs, 1, None, device, enable_acl=acl, config=config)[0]
        )

    verify(outputs, atol=atol, rtol=rtol, config=config, verify_saturation=verify_saturation)


# fmt: off
@pytest.mark.parametrize(
     "typef,dtype,size,stride,dilation,pad,ceil_mode,count_include_pad,input_shape,expected_ops",
     [
        ("nn.max_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (0, 0), False,  True, (16, 16, 16), (0, 1),),
        ("nn.max_pool2d", "float32",  (3, 3), (2, 2), (1, 1), (1, 1),  True,  True, (15, 15, 16), (0, 1),),
        ("nn.max_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16), (0, 1),),
        ("nn.max_pool2d", "uint8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16), (0, 1),),
        ("nn.max_pool2d", "uint8", (2, 2), (2, 2), (1, 1), (1, 1),  True,  True, (15, 15, 16), (0, 1),),
        ("nn.max_pool2d", "uint8", (2, 2), (2, 2), (3, 2), (1, 1),  True,  True, (15, 15, 16), (1, 0),),
        ("nn.max_pool2d", "int8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16), (0, 1),),
        ("nn.max_pool2d", "int8", (2, 2), (2, 2), (1, 1), (1, 1),  True,  True, (15, 15, 16), (0, 1),),
        ("nn.max_pool2d", "int8", (2, 2), (2, 2), (3, 2), (1, 1),  True,  True, (15, 15, 16), (1, 0),),
        ("nn.avg_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (1, 1), False, False, (16, 16, 16), (0, 1),),
        ("nn.avg_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (1, 1), False, False, (16, 16, 16), (0, 1),),
        ("nn.avg_pool2d", "float32",  (2, 2), (2, 2), (1, 1), (0, 0), False,  True, (16, 16, 16), (0, 1),),
        ("nn.avg_pool2d", "float32",  (3, 3), (2, 2), (3, 2), (0, 1),  True, False, (15, 15, 16), (1, 0),),
        ("nn.avg_pool2d", "uint8", (2, 2), (2, 2), (1, 1), (1, 1), False,  True, (16, 16, 16), (0, 1),),
        ("nn.avg_pool2d", "uint8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16), (0, 1),),
        ("nn.avg_pool2d", "int8", (2, 2), (2, 2), (1, 1), (1, 1), False,  True, (16, 16, 16), (0, 1),),
        ("nn.avg_pool2d", "int8", (3, 3), (2, 2), (1, 1), (0, 1), False, False, (16, 16, 16), (0, 1),),
        ("nn.l2_pool2d",  "float32",  (2, 2), (2, 2), (1, 1), (0, 1),  True, False, (15, 15, 16), (0, 1),),
        ("nn.l2_pool2d",  "float32",  (3, 3), (2, 2), (1, 1), (0, 0), False, False, (16, 16, 16), (0, 1),),
        ("nn.l2_pool2d",  "float32",  (2, 2), (2, 2), (1, 1), (1, 1), False,  True, (15, 15, 16), (0, 1),),
     ],
)
# fmt: on
def test_codegen_pooling(
    typef,
    dtype,
    size,
    stride,
    dilation,
    pad,
    ceil_mode,
    count_include_pad,
    input_shape,
    expected_ops,
):
    if skip_codegen_test():
        return

    low, high, _, _ = _get_low_high_atol_rtol(dtype)
    tvm_ops, acl_partitions = expected_ops

    shape = (1, *input_shape)
    inputs = {"a"}
    args = (shape, dtype, typef, size, stride, dilation, pad, False, False)
    func = _get_pooling_model(*args, iter(inputs))
    exp_codegen = _get_expected_pooling_codegen(*args)

    verify_codegen(func, exp_codegen, acl_partitions, tvm_ops)


@pytest.mark.parametrize(
    "typef,dtype,input_shape",
    [
        ("nn.global_max_pool2d", "float32", (8, 8, 16)),
        ("nn.global_max_pool2d", "float32", (9, 9, 16)),
        ("nn.global_max_pool2d", "uint8", (8, 8, 16)),
        ("nn.global_max_pool2d", "uint8", (9, 9, 16)),
        ("nn.global_max_pool2d", "int8", (8, 8, 16)),
        ("nn.global_max_pool2d", "int8", (9, 9, 16)),
        ("nn.global_avg_pool2d", "float32", (8, 8, 16)),
        ("nn.global_avg_pool2d", "float32", (9, 9, 16)),
        ("nn.global_avg_pool2d", "uint8", (8, 8, 16)),
        ("nn.global_avg_pool2d", "uint8", (9, 9, 16)),
        ("nn.global_avg_pool2d", "int8", (8, 8, 16)),
        ("nn.global_avg_pool2d", "int8", (9, 9, 16)),
    ],
)
def test_codegen_global_pooling(typef, dtype, input_shape):
    if skip_codegen_test():
        return

    low, high, _, _ = _get_low_high_atol_rtol(dtype)

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

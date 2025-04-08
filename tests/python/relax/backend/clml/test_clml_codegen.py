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
"""CLML integration operator tests."""
import pytest
import numpy as np
import tvm
import tvm.testing
import json

from tvm import relax
from tvm.script import relax as R
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder
from tvm.relax.backend.adreno import clml
from tvm.relax.backend.adreno.clml import OpenCLMLOffLoad

from mod_utils import (
    get_relax_conv2d_mod,
    get_clml_conv2d_codegen,
    get_relax_conv2d_transpose_mod,
    get_conv2d_transpose_expected_codegen,
    get_batchnorm_mod,
    get_binary_op_mod,
    get_unary_op_mod,
    get_relax_maxpool_mod,
    get_maxpool_expected_codegen,
    get_relax_avgpool_mod,
    get_avgpool_expected_codegen,
    get_relax_reshape_mod,
    get_relax_reshape_codegen,
    get_relax_global_avgpool_mod,
    get_global_avgpool_expected_codegen,
    get_relax_global_maxpool_mod,
    get_global_maxpool_expected_codegen,
)


def compare_codegen(clml_mod, clml_codegen):
    source = clml_mod.attrs["external_mods"][0].get_source()
    codegen = json.loads(source)["nodes"]
    for node in range(len(codegen)):
        if codegen[node]["op"] == "input" or codegen[node]["op"] == "const":
            codegen[node]["name"] = ""
        if codegen[node]["op"] == "kernel":
            codegen[node]["name"] = ""
    codegen_str = json.dumps(codegen, sort_keys=True, indent=2)
    known_good_codegen_str = json.dumps(clml_codegen, sort_keys=True, indent=2)
    assert codegen_str == known_good_codegen_str, (
        f"The JSON produced by codegen does not match the expected result. \n"
        f"Actual={codegen_str} \n"
        f"Expected={known_good_codegen_str}"
    )


def verify(mod, params_np, clml_codegen):
    mod = tvm.relax.transform.BindParams("main", params_np)(mod)
    clml_mod = OpenCLMLOffLoad()(mod)
    compare_codegen(clml_mod, clml_codegen)


@tvm.testing.requires_openclml
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "kernel_h, kernel_w, padding, stride, dilation, out_channels, shape, has_bias, has_bn, has_activation, has_pad, is_depthwise",
    [
        (3, 3, (1, 1), (1, 1), (1, 1), 64, (3, 224, 224), False, True, False, True, False),
        (3, 3, (1, 1), (1, 1), (1, 1), 64, (3, 224, 224), False, True, False, False, False),
        (5, 5, (2, 2), (1, 1), (1, 1), 16, (16, 64, 64), False, True, True, False, False),
        (7, 7, (3, 3), (2, 2), (1, 1), 32, (3, 224, 224), True, False, True, True, False),
        (3, 3, (0, 0), (1, 1), (1, 1), 512, (256, 14, 14), True, False, True, False, False),
        (1, 1, (0, 0), (1, 1), (1, 1), 1024, (512, 7, 7), True, False, True, False, False),
        (1, 3, (0, 0), (1, 1), (1, 1), 64, (64, 7, 7), True, False, True, False, False),
        (3, 1, (0, 0), (1, 1), (1, 1), 64, (64, 7, 7), False, True, True, True, False),
    ],
)
def test_conv2d_offload(
    kernel_h,
    kernel_w,
    padding,
    stride,
    dilation,
    out_channels,
    shape,
    has_bias,
    has_bn,
    has_activation,
    has_pad,
    is_depthwise,
    dtype,
):
    low, high = 0, 1
    data_shape = (1, *shape)
    if is_depthwise:
        groups = data_shape[1] // out_channels
    else:
        groups = 1
    padding = (padding[0], padding[1], padding[0], padding[1])

    weight_format = "IOHW" if is_depthwise else "OIHW"
    weight_shape = (out_channels, data_shape[1] // groups, kernel_h, kernel_w)

    weight = np.random.uniform(low, high, size=weight_shape).astype(dtype)
    bias = np.random.uniform(low, high, size=(1, weight_shape[0], 1, 1)).astype(dtype)

    gamma = np.random.uniform(low, high, size=(weight_shape[0],)).astype(dtype)
    beta = np.random.uniform(low, high, size=(weight_shape[0],)).astype(dtype)
    mean = np.random.uniform(low, high, size=(weight_shape[0],)).astype(dtype)
    variance = np.random.uniform(low, high, size=(weight_shape[0],)).astype(dtype)

    params_np = {"weight": weight}
    if has_bias:
        params_np["bias"] = bias
    if has_bn:
        params_np.update({"gamma": gamma, "beta": beta, "mean": mean, "variance": variance})

    mod = get_relax_conv2d_mod(
        data_shape,
        weight_shape,
        stride=stride,
        dilation=dilation,
        padding=padding,
        weight_layout=weight_format,
        groups=groups,
        dtype=dtype,
        has_bias=has_bias,
        has_bn=has_bn,
        has_activation=has_activation,
        has_pad=has_pad,
        is_depthwise=is_depthwise,
    )

    clml_codegen = get_clml_conv2d_codegen(
        data_shape,
        weight_shape,
        stride=stride,
        dilation=dilation,
        padding=padding,
        weight_layout=weight_format,
        groups=groups,
        dtype=dtype,
        has_bias=has_bias,
        has_bn=has_bn,
        has_activation=has_activation,
        has_pad=has_pad,
        is_depthwise=is_depthwise,
    )

    verify(mod, params_np, clml_codegen)


@tvm.testing.requires_openclml
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "dshape, kshape, channels, kernel_size, strides, padding, out_shape",
    [
        ((1, 256, 100, 100), (64, 256, 4, 4), 64, (4, 4), (2, 2), (0, 0, 0, 0), (1, 64, 202, 202)),
        ((1, 64, 200, 200), (64, 64, 4, 4), 64, (4, 4), (2, 2), (1, 1, 1, 1), (1, 64, 400, 400)),
        ((1, 64, 200, 200), (64, 64, 4, 4), 64, (4, 4), (2, 2), (1, 1, 1, 1), (1, 64, 400, 400)),
        ((1, 64, 400, 400), (16, 64, 4, 4), 16, (4, 4), (2, 2), (1, 1, 1, 1), (1, 16, 800, 800)),
    ],
)
def test_conv2d_transpose(
    dshape, kshape, channels, kernel_size, strides, padding, dtype, out_shape
):
    low, high = -1, 1
    weight = np.random.uniform(low, high, size=kshape).astype(dtype)

    params_np = {"weight": weight}

    mod = get_relax_conv2d_transpose_mod(
        dshape,
        kshape,
        channels=channels,
        stride=strides,
        padding=padding,
        dtype=dtype,
    )

    exp_codegen = get_conv2d_transpose_expected_codegen(
        dshape=dshape,
        kshape=kshape,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=(1, 1),
        dtype=dtype,
        output_shape=out_shape,
    )
    verify(mod, params_np, exp_codegen)


@tvm.testing.requires_openclml
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 64, 14, 14), 1, 3e-4],
        [(1, 14, 256, 256), 1, 3e-4],
        [(1, 14, 256, 256), 1, 3e-4],
        [(1, 256, 1, 1), 1, 3e-4],
    ],
)
def test_batchnorm(dtype, trials):
    low, high = 0, 1
    if clml.clml_sdk_version() < 3:
        print("Skip due to unsupported CLML version:", clml.clml_sdk_version())
        return

    (input_shape, axis, epsilon) = trials
    channels = input_shape[axis]

    def _get_axis_tuple(axis):
        if axis == 0:
            return (1, 2, 3)
        elif axis == 1:
            return (0, 2, 3)
        elif axis == 2:
            return (0, 1, 3)
        else:
            return (0, 1, 2)

    data = np.random.uniform(low, high, size=(input_shape)).astype(dtype)
    gamma = np.random.uniform(low, high, size=(channels)).astype(dtype)
    beta = np.random.uniform(low, high, size=(channels)).astype(dtype)
    mean = np.mean(data, _get_axis_tuple(axis), keepdims=False)
    variance = np.var(data, _get_axis_tuple(axis), keepdims=False)

    params_np = {"gamma": gamma, "beta": beta, "moving_mean": mean, "moving_var": variance}
    mod = get_batchnorm_mod(input_shape, channels, axis, epsilon, dtype)
    exp_codegen = [
        {
            "attrs": {"dtype": [[dtype]], "shape": [[input_shape]]},
            "name": "",
            "op": "input",
        },
        {"attrs": {"dtype": [[dtype]], "shape": [[[channels]]]}, "name": "", "op": "const"},
        {"attrs": {"dtype": [[dtype]], "shape": [[[channels]]]}, "name": "", "op": "const"},
        {"attrs": {"dtype": [[dtype]], "shape": [[[channels]]]}, "name": "", "op": "const"},
        {"attrs": {"dtype": [[dtype]], "shape": [[[channels]]]}, "name": "", "op": "const"},
        {
            "attrs": {
                "axis": [[str(axis)]],
                "center": [["1"]],
                "dtype": [[dtype]],
                "clml_version": [["3"]],
                "momentum": [["0.10000000000000001"]],
                "epsilon": [["0.00029999999999999997"]],
                "num_inputs": "5",
                "num_outputs": "1",
                "scale": [["1"]],
                "shape": [[input_shape]],
            },
            "inputs": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            "name": "",
            "op": "kernel",
        },
    ]
    verify(mod, params_np, exp_codegen)


@tvm.testing.requires_openclml
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "a_shape, b_shape, op",
    [
        ((1, 64, 14, 14), (1, 64, 14, 14), R.add),
        ((1, 256), (1, 256), R.add),
        ((1, 64, 14, 14), (1, 64, 14, 14), R.subtract),
        ((1, 256), (1, 256), R.subtract),
        ((1, 64, 14, 14), (1, 64, 14, 14), R.multiply),
        ((1, 256), (1, 256), R.multiply),
        ((1, 64, 14, 14), (1, 64, 14, 14), R.divide),
        ((1, 256), (1, 256), R.divide),
        ((1, 64, 14, 14), (1, 64, 14, 14), R.minimum),
        ((1, 256), (1, 256), R.minimum),
        ((1, 64, 14, 14), (1, 64, 14, 14), R.maximum),
        ((1, 256), (1, 256), R.maximum),
    ],
)
@tvm.testing.requires_openclml
def test_binary_ops(a_shape, b_shape, op, dtype):
    def _verify(mod):
        expected_codegen_str = [
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "shape": [[a_shape]],
                },
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "shape": [[b_shape]],
                },
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "clml_version": [["3"]],
                    "dtype": [[dtype]],
                    "num_inputs": "2",
                    "num_outputs": "1",
                    "shape": [[a_shape]],
                },
                "inputs": [[0, 0, 0], [1, 0, 0]],
                "name": "",
                "op": "kernel",
            },
        ]
        verify(mod, {}, expected_codegen_str)

    (mod, _) = get_binary_op_mod(a_shape, b_shape, op, dtype)

    _verify(mod)


@tvm.testing.requires_openclml
@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
    ],
)
@pytest.mark.parametrize(
    "a_shape, op",
    [
        ((1, 64, 14, 14), R.nn.relu),
        ((1, 256, 1, 1), R.nn.relu),
        ((1, 14, 256, 256), R.nn.relu),
        ((1, 14, 14, 256), R.nn.relu),
    ],
)
@tvm.testing.requires_openclml
def test_unary_ops(a_shape, op, dtype):
    def _verify(mod):
        expected_codegen_str = [
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "shape": [[a_shape]],
                },
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "activation_type": [["relu"]],
                    "clml_version": [["3"]],
                    "dtype": [[dtype]],
                    "num_inputs": "1",
                    "num_outputs": "1",
                    "shape": [[a_shape]],
                },
                "inputs": [[0, 0, 0]],
                "name": "",
                "op": "kernel",
            },
        ]
        verify(mod, {}, expected_codegen_str)

    (mod, _) = get_unary_op_mod(a_shape, op, dtype)

    _verify(mod)


@tvm.testing.requires_openclml
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 64, 147, 147), (3, 3), (2, 2), (1, 1), (0, 0, 0, 0), False],
        [(1, 256, 17, 17), (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), False],
        [(1, 1024, 14, 14), (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), False],
        [(1, 32, 256, 256), (3, 3), (2, 2), (1, 1), (1, 1, 1, 1), True],
        [(1, 32, 256, 256), (3, 3), (2, 2), (1, 1), (0, 1, 0, 1), True],
        [(1, 32, 256, 256), (2, 2), (2, 2), (1, 1), (1, 1, 1, 1), True],
        [(1, 32, 256, 256), (2, 2), (2, 2), (1, 1), (1, 0, 1, 0), True],
    ],
)
def test_max_pool(dtype, trials):
    low, high = -1, 1
    (input_shape, pool_size, stride, dilation, padding, has_pad) = trials
    mod = get_relax_maxpool_mod(input_shape, dtype, pool_size, stride, dilation, padding, has_pad)
    params_np = {}

    expected_codegen_str = get_maxpool_expected_codegen(
        input_shape, pool_size, stride, padding, "maxpool2d", dtype
    )
    verify(mod, params_np, expected_codegen_str)


@tvm.testing.requires_openclml
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 64, 147, 147), (3, 3), (2, 2), (1, 1), (0, 0, 0, 0), False],
        [(1, 256, 17, 17), (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), False],
        [(1, 1024, 14, 14), (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), False],
        [(1, 32, 256, 256), (3, 3), (2, 2), (1, 1), (1, 1, 1, 1), True],
        [(1, 32, 256, 256), (3, 3), (2, 2), (1, 1), (0, 1, 0, 1), True],
        [(1, 32, 256, 256), (2, 2), (2, 2), (1, 1), (1, 1, 1, 1), True],
        [(1, 32, 256, 256), (2, 2), (2, 2), (1, 1), (1, 0, 1, 0), True],
    ],
)
def test_avg_pool(dtype, trials):
    low, high = -1, 1
    (input_shape, pool_size, stride, dilation, padding, has_pad) = trials
    mod = get_relax_avgpool_mod(input_shape, dtype, pool_size, stride, dilation, padding, has_pad)
    params_np = {}
    exp_codegen_str = get_avgpool_expected_codegen(
        input_shape, pool_size, stride, padding, "avg_pool2d", dtype
    )
    verify(mod, params_np, exp_codegen_str)


@tvm.testing.requires_openclml
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 3, 32, 32), (1, 4, -1, 32)],
        [(1, 4, 8, 32), (1, 4, -1, 16)],
        [(1, 64, 3, 3), (1, 32, 3, -1)],
    ],
)
def test_reshape(dtype, trials):
    low, high = -1, 1
    (input_shape, output_shape) = trials
    mod = get_relax_reshape_mod(input_shape, output_shape, dtype)
    params_np = {}
    expected_codegen = get_relax_reshape_codegen(input_shape, output_shape, dtype)
    verify(mod, params_np, expected_codegen)


@tvm.testing.requires_openclml
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 64, 147, 147), True],
        [(1, 256, 17, 17), False],
        [(1, 1024, 14, 14), True],
        [(1, 32, 256, 256), False],
    ],
)
def test_global_avg_pool(dtype, trials):
    """Test function for global average pooling."""
    low, high = -1, 1
    (input_shape, keep_dims) = trials
    mod = get_relax_global_avgpool_mod(input_shape, keep_dims, dtype)
    params_np = {}
    exp_codegen_str = get_global_avgpool_expected_codegen(input_shape, keep_dims, dtype)
    verify(mod, params_np, exp_codegen_str)


@tvm.testing.requires_openclml
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 64, 147, 147), True],
        [(1, 256, 17, 17), False],
        [(1, 1024, 14, 14), True],
        [(1, 32, 256, 256), False],
    ],
)
def test_global_max_pool(dtype, trials):
    """Test function for global average pooling."""
    low, high = -1, 1
    (input_shape, keep_dims) = trials
    N, C, H, W = input_shape
    pool_size = (H, W)
    stride = (1, 1)
    padding = (0, 0, 0, 0)
    mod = get_relax_global_maxpool_mod(input_shape, keep_dims, dtype)
    params_np = {}
    exp_codegen_str = get_global_maxpool_expected_codegen(
        input_shape, pool_size, stride, padding, "global_max", dtype
    )
    verify(mod, params_np, exp_codegen_str)


if __name__ == "__main__":
    tvm.testing.main()

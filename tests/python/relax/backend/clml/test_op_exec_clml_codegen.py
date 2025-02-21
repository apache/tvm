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

from tvm import relax, rpc
from tvm.script import relax as R
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder
from tvm.relax.backend.adreno import clml
from utils import run_compare

from mod_utils import (
    get_relax_conv2d_mod,
    get_batchnorm_mod,
    get_binary_op_mod,
    get_unary_op_mod,
    get_relax_maxpool_mod,
    get_relax_avgpool_mod,
    get_relax_reshape_mod,
    get_relax_reshape_codegen,
    get_relax_global_avgpool_mod,
    get_relax_global_maxpool_mod,
)


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
    rpc,
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

    data = np.random.uniform(low, high, size=data_shape).astype(dtype)
    weight = np.random.uniform(low, high, size=weight_shape).astype(dtype)
    bias = np.random.uniform(low, high, size=(1, weight_shape[0], 1, 1)).astype(dtype)

    gamma = np.random.uniform(low, high, size=(weight_shape[0],)).astype(dtype)
    beta = np.random.uniform(low, high, size=(weight_shape[0],)).astype(dtype)
    mean = np.random.uniform(low, high, size=(weight_shape[0],)).astype(dtype)
    variance = np.random.uniform(low, high, size=(weight_shape[0],)).astype(dtype)

    inputs = [data]
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
    run_compare(mod, inputs, params_np, rpc)


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
def test_batchnorm(dtype, trials, rpc):
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

    inputs = [data]
    params_np = {"gamma": gamma, "beta": beta, "moving_mean": mean, "moving_var": variance}
    mod = get_batchnorm_mod(input_shape, channels, axis, epsilon, dtype)
    run_compare(mod, inputs, params_np, rpc)


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
def test_binary_ops(a_shape, b_shape, op, rpc, dtype):
    (mod, inputs) = get_binary_op_mod(a_shape, b_shape, op, dtype)
    run_compare(mod, inputs, {}, rpc)


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
def test_unary_ops(a_shape, op, rpc, dtype):
    (mod, inputs) = get_unary_op_mod(a_shape, op, dtype)
    run_compare(mod, inputs, {}, rpc)


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
def test_max_pool(dtype, trials, rpc):
    low, high = -1, 1
    (input_shape, pool_size, stride, dilation, padding, has_pad) = trials
    data = np.random.uniform(low, high, size=input_shape).astype(dtype)
    inputs = [data]
    mod = get_relax_maxpool_mod(input_shape, dtype, pool_size, stride, dilation, padding, has_pad)
    params_np = {}
    run_compare(mod, inputs, params_np, rpc)


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
def test_avg_pool(dtype, trials, rpc):
    low, high = -1, 1
    (input_shape, pool_size, stride, dilation, padding, has_pad) = trials
    data = np.random.uniform(low, high, size=input_shape).astype(dtype)
    inputs = [data]
    mod = get_relax_avgpool_mod(input_shape, dtype, pool_size, stride, dilation, padding, has_pad)
    params_np = {}
    run_compare(mod, inputs, params_np, rpc)


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
def test_reshape(dtype, trials, rpc):
    low, high = -1, 1
    (input_shape, output_shape) = trials
    data = np.random.uniform(low, high, size=input_shape).astype(dtype)
    inputs = [data]
    mod = get_relax_reshape_mod(input_shape, output_shape, dtype)
    params_np = {}
    run_compare(mod, inputs, params_np, rpc)


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
def test_global_avg_pool(dtype, trials, rpc):
    """Test function for global average pooling."""
    low, high = -1, 1
    (input_shape, keep_dims) = trials
    data = np.random.uniform(low, high, size=input_shape).astype(dtype)
    inputs = [data]
    mod = get_relax_global_avgpool_mod(input_shape, keep_dims, dtype)
    params_np = {}
    run_compare(mod, inputs, params_np, rpc)


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
def test_global_max_pool(dtype, trials, rpc):
    """Test function for global average pooling."""
    low, high = -1, 1
    (input_shape, keep_dims) = trials
    N, C, H, W = input_shape
    pool_size = (H, W)
    stride = (1, 1)
    padding = (0, 0, 0, 0)
    data = np.random.uniform(low, high, size=input_shape).astype(dtype)
    inputs = [data]
    mod = get_relax_global_maxpool_mod(input_shape, keep_dims, dtype)
    params_np = {}
    run_compare(mod, inputs, params_np, rpc)


if __name__ == "__main__":
    tvm.testing.main()

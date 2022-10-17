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
"""Example code to do convolution."""

import sys

import pytest
import numpy as np

import tvm
from tvm import autotvm, te, topi
import tvm.topi.testing
from tvm.contrib import cudnn
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.conv2d import _get_workload
from tvm.topi.x86.conv2d_avx_common import _fallback_schedule

import tvm.testing

dtype = tvm.testing.parameter("float16", "float32")
random_seed = tvm.testing.parameter(0)


@tvm.testing.fixture
def input_shape(batch, in_channel, in_size):
    return (batch, in_channel, in_size, in_size)


@tvm.testing.fixture
def weight_shape(num_filter, in_channel, kernel):
    return (num_filter, in_channel, kernel, kernel)


@tvm.testing.fixture
def bias_shape(num_filter):
    return (num_filter, 1, 1)


@tvm.testing.fixture(cache_return_value=True)
def ref_data(
    random_seed,
    input_shape,
    weight_shape,
    bias_shape,
    dtype,
    stride,
    padding,
    dilation,
    add_bias,
    apply_relu,
):
    np.random.seed(random_seed)

    # scipy.signal.convolve2d does not support float16 data types, and
    # the python fallback is too slow for general use.  Computing
    # ref_data in float32 will have fewer rounding errors than the TVM
    # float16 compute, but those vary based on schedule anyways.
    conv_dtype = "float32" if dtype == "float16" else dtype

    a_np = np.random.uniform(size=input_shape).astype(dtype)
    w_np = np.random.uniform(size=weight_shape).astype(dtype)
    b_np = np.random.uniform(size=bias_shape).astype(dtype)
    dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
    c_np = tvm.topi.testing.conv2d_nchw_python(
        a_np.astype(conv_dtype), dw_np.astype(conv_dtype), stride, padding
    ).astype(dtype)

    if add_bias:
        c_np = c_np + b_np
    if apply_relu:
        c_np = np.maximum(c_np, 0)
    return a_np, w_np, b_np, c_np


class BaseConv2DTests:
    add_bias = tvm.testing.parameter(False)
    apply_relu = tvm.testing.parameter(False)
    dilation = tvm.testing.parameter(1)
    batch = tvm.testing.parameter(1)

    def test_conv2d_nchw(
        self,
        target,
        dev,
        batch,
        in_channel,
        in_size,
        num_filter,
        kernel,
        stride,
        padding,
        dtype,
        ref_data,
        dilation,
        add_bias,
        apply_relu,
    ):
        target = tvm.target.Target(target)
        is_cudnn_target = target.kind.name == "cuda" and "cudnn" in target.attrs.get("libs", [])

        if target.kind.name == "vulkan" and dtype == "float16":
            if not target.attrs.get("supports_float16", False) or not target.attrs.get(
                "supports_16bit_buffer", False
            ):
                pytest.xfail("Vulkan device does not support float16")

        if (
            target.kind.name == "cuda"
            and dtype == "float16"
            and not tvm.contrib.nvcc.have_fp16(dev.compute_version)
        ):
            pytest.xfail("CUDA float16 intrinsics not available")

        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
        padding_sum = pad_top + pad_left + pad_bottom + pad_right

        has_asymmetric_padding = (pad_top != pad_bottom) or (pad_left != pad_right)
        if is_cudnn_target and has_asymmetric_padding:
            pytest.xfail("CuDNN does not support asymmetric padding")

        a_np, w_np, b_np, c_np = ref_data

        A = te.placeholder(a_np.shape, name="A", dtype=dtype)
        W = te.placeholder(w_np.shape, name="W", dtype=dtype)
        bias = te.placeholder(b_np.shape, name="bias", dtype=dtype)

        if "int" in dtype:
            tol = {"atol": 0, "rtol": 0}
        elif dtype == "float32":
            tol = {"rtol": 1e-4, "atol": 2e-4}
        elif dtype == "float16":
            # A summation in float16 with a single accumulator very
            # quickly runs into large rounding errors.  At some point,
            # this tolerance should be schedule-dependent for to avoid
            # false negatives.
            num_values_summed = in_channel * kernel * kernel
            gap_size = np.nextafter(c_np.max(), np.inf, dtype=c_np.dtype) - c_np.max()
            tol = {"rtol": 1e-3, "atol": num_values_summed * gap_size / 2}

        with autotvm.tophub.context(target):  # load tophub pre-tuned parameters
            if is_cudnn_target:
                fcompute, fschedule = topi.cuda.conv2d_cudnn, topi.cuda.schedule_conv2d_cudnn
            else:
                fcompute, fschedule = tvm.topi.testing.get_conv2d_nchw_implement(target)

            with target:
                if is_cudnn_target:
                    C = fcompute(
                        A, W, (stride, stride), padding, (dilation, dilation), 1, "NCHW", dtype
                    )
                else:
                    C = fcompute(A, W, (stride, stride), padding, (dilation, dilation), dtype)
                if add_bias:
                    C = topi.add(C, bias)
                if apply_relu:
                    C = topi.nn.relu(C)
                s = fschedule([C])

            a = tvm.nd.array(a_np, dev)
            w = tvm.nd.array(w_np, dev)
            b = tvm.nd.array(b_np, dev)

            c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
            func = tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="conv2d_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                    dtype,
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    kernel,
                    stride,
                    padding_sum,
                    dilation,
                ),
            )
            func(a, w, b, c)
            tvm.testing.assert_allclose(c.numpy(), c_np, **tol)

    @tvm.testing.parametrize_targets("llvm")
    def test_workload_padding(
        self,
        target,
        input_shape,
        weight_shape,
        stride,
        padding,
        dilation,
        dtype,
        ref_data,
    ):
        a_np, w_np, b_np, c_np = ref_data
        _, _, out_height, out_width = c_np.shape

        A = te.placeholder(input_shape, name="A", dtype=dtype)
        W = te.placeholder(weight_shape, name="W", dtype=dtype)

        with tvm.target.Target(target):
            wkl = _get_workload(A, W, (stride, stride), padding, dilation, dtype)

            # check if tile_ow candidates are the factors of the right output weight.
            cfg = autotvm.get_config()
            _fallback_schedule(cfg, wkl)
            ow_tile = np.prod(cfg["tile_ow"].size)

        tvm.testing.assert_allclose(ow_tile, out_width)


class TestResNet18Workloads(BaseConv2DTests):
    in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (3, 224, 64, 7, 2, 3),
        (64, 56, 64, 3, 1, 1),
        (64, 56, 64, 1, 1, 0),
        (64, 56, 128, 3, 2, 1),
        (64, 56, 128, 1, 2, 0),
        (128, 28, 128, 3, 1, 1),
        (128, 28, 256, 3, 2, 1),
        (128, 28, 256, 1, 2, 0),
        (256, 14, 256, 3, 1, 1),
        (256, 14, 512, 3, 2, 1),
        (256, 14, 512, 1, 2, 0),
        (512, 7, 512, 3, 1, 1),
    )


class TestInceptionV3Workloads(BaseConv2DTests):
    in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (3, 299, 32, 3, 2, 0),
        (32, 149, 32, 3, 1, 0),
        (32, 147, 64, 3, 1, 1),
        (64, 73, 80, 1, 1, 0),
        (80, 73, 192, 3, 1, 0),
        (192, 35, 64, 1, 1, 0),
        (192, 35, 48, 1, 1, 0),
        (48, 35, 64, 5, 1, 2),
        (64, 35, 96, 3, 1, 1),
        (96, 35, 96, 3, 1, 1),
        (192, 35, 32, 1, 1, 0),
        (256, 35, 64, 1, 1, 0),
        (256, 35, 48, 1, 1, 0),
        (288, 35, 64, 1, 1, 0),
        (288, 35, 48, 1, 1, 0),
        (288, 35, 384, 3, 2, 0),
        (96, 35, 96, 3, 2, 0),
        (768, 17, 192, 1, 1, 0),
        (768, 17, 128, 1, 1, 0),
        (128, 17, 128, 1, 1, 0),
        (128, 17, 192, 7, 1, 3),
        (128, 17, 128, 7, 1, 3),
        (128, 17, 192, 1, 1, 0),
        (768, 17, 160, 1, 1, 0),
        # disable these tests due to some bugs of llvm with nvptx
        # (160,  17, 160, 1, 1, 0),
        (160, 17, 192, 7, 1, 3),
        (160, 17, 160, 7, 1, 3),
        (160, 17, 192, 1, 1, 0),
        (192, 17, 192, 1, 1, 0),
        (192, 17, 192, 7, 1, 3),
        (192, 17, 320, 3, 2, 0),
        (192, 17, 192, 3, 2, 0),
        (1280, 8, 320, 1, 1, 0),
        (1280, 8, 384, 1, 1, 0),
        (384, 8, 384, 1, 1, 0),
        (384, 8, 384, 3, 1, 1),
        (1280, 8, 448, 1, 1, 0),
        (448, 8, 384, 3, 1, 1),
        (1280, 8, 192, 1, 1, 0),
        (2048, 8, 320, 1, 1, 0),
        (2048, 8, 384, 1, 1, 0),
        (2048, 8, 448, 1, 1, 0),
        (2048, 8, 192, 1, 1, 0),
        (1024, 19, 84, 3, 1, 1),
        (2048, 10, 126, 3, 1, 1),
        (512, 5, 126, 3, 1, 1),
        (256, 3, 126, 3, 1, 1),
    )


class TestWeirdWorkloads(BaseConv2DTests):
    batch, in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (2, 2, 2, 2, 2, 2, 2),
        (3, 3, 3, 3, 3, 3, 3),
        (4, 4, 4, 4, 4, 4, 4),
        (5, 5, 5, 5, 5, 5, 5),
        (6, 6, 6, 6, 6, 6, 6),
        # disable these tests due to some bugs of llvm with nvptx
        # (1, 1, 1, 1, 1, 1, 1),
        # (2, 13, 71, 59, 3, 1, 1),
    )


class TestAsymmetricPadding(BaseConv2DTests):
    dilation = tvm.testing.parameter(1, 2)
    in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (3, 35, 64, 7, 2, (0, 0, 1, 1)),
        (64, 8, 128, 3, 1, (3, 3, 2, 2)),
        (64, 8, 64, 1, 1, (1, 2, 2, 1)),
        (64, 17, 192, 1, 1, (1, 2)),
        (64, 8, 64, 3, 1, (3, 1)),
        (128, 8, 384, 3, 1, (0, 2)),
        (64, 35, 64, 3, 1, (1, 2)),
        (64, 8, 64, 1, 1, "VALID"),
        (388, 8, 64, 3, 1, "VALID"),
        (64, 10, 48, 3, 1, "VALID"),
        (512, 19, 64, 1, 1, "SAME"),
        (64, 5, 32, 2, 1, "SAME"),
        (64, 8, 64, 3, 1, "SAME"),
        (64, 8, 64, 3, 1, (1, 2, 2, 1)),
        (64, 8, 64, 5, 2, (1, 3)),
        (64, 8, 64, 3, 1, "VALID"),
        (64, 8, 64, 24, 1, "SAME"),
        (32, 35, 64, 7, 2, (0, 0, 2, 2)),
    )


class TestBatchSize(BaseConv2DTests):
    in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (64, 56, 64, 3, 1, 1),
    )
    batch = tvm.testing.parameter(1, 4, 9)


class TestBiasRelu(BaseConv2DTests):
    apply_relu = tvm.testing.parameter(True, False, ids=["relu", "no_relu"])
    add_bias = tvm.testing.parameter(True, False, ids=["bias", "no_bias"])
    in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (64, 56, 64, 3, 1, 1),
        (64, 8, 64, 3, 1, (1, 2, 2, 1)),
        (64, 8, 64, 5, 2, (1, 3)),
        (64, 8, 64, 3, 1, "VALID"),
        (64, 8, 64, 24, 1, "SAME"),
    )


if __name__ == "__main__":
    tvm.testing.main()

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
#
"""Example code to do convolution."""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.conv2d import _get_workload
from tvm.topi.generic.conv2d import fallback_schedule_cpu_common_int8

from common import Int8Fallback
import tvm.testing
import pytest
import platform


def compile_conv2d_NHWC_gemm_int8_arm(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation=1,
    add_bias=False,
    add_relu=False,
):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation)
    )

    in_height = in_width = in_size
    A = te.placeholder((batch, in_height, in_width, in_channel), name="A", dtype="int8")
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W", dtype="int8")
    bias = te.placeholder((num_filter,), name="bias", dtype="int8")
    dtype = "int32"
    devices = [
        (
            "llvm --device arm_cpu --mtriple aarch64-linux-gnu",
            topi.arm_cpu.compute_conv2d_NHWC_quantized_interleaved,
            topi.arm_cpu.schedule_conv2d_NHWC_quantized_interleaved,
        ),
        (
            "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v8.2a,+dotprod",
            topi.arm_cpu.compute_conv2d_NHWC_quantized_interleaved,
            topi.arm_cpu.schedule_conv2d_NHWC_quantized_interleaved,
        ),
        (
            "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v8.2a,+dotprod",
            topi.arm_cpu.compute_conv2d_NHWC_quantized_native,
            topi.arm_cpu.schedule_conv2d_NHWC_quantized_native,
        ),
        # TODO(giuseros) Need LLVM-11 in order to compile with +i8mm extension
        # (
        #   "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v8.2a,+i8mm",
        #   topi.arm_cpu.compute_conv2d_NHWC_quantized_interleaved,
        #   topi.arm_cpu.schedule_conv2d_NHWC_quantized_interleaved,
        # ),
    ]

    for device_tuple in devices:
        target = device_tuple[0]
        compute = device_tuple[1]
        schedule = device_tuple[2]

        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Compiling on arm AArch64 target: %s" % target)
        with tvm.target.Target(target) as tvm_target:
            assert tvm_target.features.is_aarch64, "AArch64 target not recognized"

            C = compute(A, W, (stride, stride), padding, (dilation, dilation), dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = schedule([C])

        if add_bias:
            tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )
            func = tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="relu_%dnnn_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )
        else:
            func = tvm.build(
                s,
                [A, W, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )


def verify_conv2d_NHWC_gemm_int8(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation=1,
    add_bias=False,
    add_relu=False,
):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation)
    )

    in_height = in_width = in_size

    A = te.placeholder((batch, in_height, in_width, in_channel), name="A", dtype="int8")
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W", dtype="int8")
    bias = te.placeholder((num_filter,), name="bias", dtype="int8")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_int8.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.randint(low=-128, high=127, size=a_shape).astype(dtype)
        w_np = np.random.randint(low=-128, high=128, size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        c_np = tvm.topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding).astype(dtype)

        if add_bias:
            b_np = np.random.uniform(size=bias_shape).astype(dtype)
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)

        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_target(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            C = topi.arm_cpu.compute_conv2d_NHWC_quantized_interleaved(
                A, W, (stride, stride), padding, (dilation, dilation), dtype
            )
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.arm_cpu.schedule_conv2d_NHWC_quantized_interleaved([C])

        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
        if add_bias:
            tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )
            func = tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )
            func(a, w, b, c)
        else:
            func = tvm.build(
                s,
                [A, W, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )
            func(a, w, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    check_target("llvm")


def verify_conv2d_NCHWc_int8(
    in_dtype,
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation=1,
    add_bias=False,
    add_relu=False,
):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation)
    )

    in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A", dtype=in_dtype)
    W = te.placeholder((num_filter, in_channel, kernel, kernel), name="W", dtype=in_dtype)

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype
    out_dtype = "int32" if in_dtype == "int8" else "uint32"
    lo = -128 if in_dtype == "int8" else 0
    hi = 127 if in_dtype == "int8" else 255

    def check_target(target, compute, schedule, oc_block_factor, build_only):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        if target == "cuda" and not tvm.contrib.nvcc.have_int8(dev.compute_version):
            print("Skip because int8 intrinsics are not available")
            return

        bias = te.placeholder(
            (num_filter // oc_block_factor, 1, 1, oc_block_factor), name="bias", dtype=out_dtype
        )
        bias_shape = get_const_tuple(bias.shape)

        @memoize("topi.tests.test_topi_conv2d_int8.verify_conv2d_nchw")
        def get_ref_data():
            a_np = np.random.randint(low=lo, high=hi, size=a_shape).astype(out_dtype)
            w_np = np.random.randint(low=lo, high=hi, size=w_shape).astype(out_dtype)
            b_np = np.random.uniform(size=bias_shape).astype(out_dtype)
            dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
            c_np = tvm.topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding).astype(
                out_dtype
            )

            # convert to NCHWc
            _, _, out_height, out_width = c_np.shape
            c_np = c_np.reshape(
                (batch, num_filter // oc_block_factor, oc_block_factor, out_height, out_width)
            ).transpose(0, 1, 3, 4, 2)

            if add_bias:
                b_np = np.random.uniform(size=bias_shape).astype(out_dtype)
                c_np += b_np
            if add_relu:
                c_np = np.maximum(c_np, 0)

            return a_np, w_np, b_np, c_np

        a_np, w_np, b_np, c_np = get_ref_data()

        with tvm.target.Target(target):
            C = compute(
                A,
                W,
                (stride, stride),
                padding,
                (dilation, dilation),
                "NCHW",
                "NCHW",
                out_dtype,
            )
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = schedule([C])

        a = tvm.nd.array(a_np.astype(dtype), dev)
        w = tvm.nd.array(w_np.astype(dtype), dev)
        b = tvm.nd.array(b_np.astype(out_dtype), dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)

        if add_bias:
            compile_args = [A, W, bias, C]
            run_args = [a, w, b, c]
        else:
            compile_args = [A, W, C]
            run_args = [a, w, c]

        func = tvm.build(
            s,
            compile_args,
            target,
            name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
            % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
        )

        if build_only:
            return

        print("Running on target: %s" % target)

        func(*run_args)

        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    targets = [
        (
            "cuda",
            lambda a, w, s, p, d, l, ol, o: topi.cuda.conv2d_NCHWc_int8(a, w, s, p, d, l, o),
            topi.cuda.schedule_conv2d_NCHWc_int8,
            4,
            False,
        ),
        # Disable on CI since it does not support spirv int8 dot product
        # (
        #     "vulkan -from_device=0",
        #     lambda a, w, s, p, d, l, ol, o: topi.cuda.conv2d_NCHWc_int8(a, w, s, p, d, l, o),
        #     topi.cuda.schedule_conv2d_NCHWc_int8,
        #     4,
        #     False,
        # ),
    ]

    build_only_aarch64 = platform.machine() != "aarch64"

    targets.append(
        (
            "llvm -device arm_cpu -mtriple aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod",
            topi.arm_cpu.conv2d_NCHWc_int8,
            topi.arm_cpu.schedule_conv2d_NCHWc_int8,
            8,
            build_only_aarch64,
        )
    )

    if in_dtype == "int8":
        targets += [
            (
                "llvm -device arm_cpu -mtriple aarch64-linux-gnu -mattr=+neon",
                topi.arm_cpu.conv2d_NCHWc_int8,
                topi.arm_cpu.schedule_conv2d_NCHWc_int8,
                8,
                build_only_aarch64,
            ),
            (
                "rocm -mattr=+dotprod",
                lambda a, w, s, p, d, l, ol, o: topi.cuda.conv2d_NCHWc_int8(a, w, s, p, d, l, o),
                topi.cuda.schedule_conv2d_NCHWc_int8,
                4,
                False,
            ),
        ]

    for target, compute, schedule, oc_block_factor, build_only in targets:
        check_target(target, compute, schedule, oc_block_factor, build_only)


def verify_conv2d_nchw_int8(
    in_dtype,
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation=1,
    add_bias=False,
    add_relu=False,
):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation)
    )

    in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A", dtype=in_dtype)
    W = te.placeholder((num_filter, in_channel, kernel, kernel), name="W", dtype=in_dtype)
    bias = te.placeholder((num_filter, 1, 1), name="bias", dtype=in_dtype)

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_int8.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.randint(low=-128, high=127, size=a_shape).astype(dtype)
        w_np = np.random.randint(low=-128, high=128, size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = tvm.topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding).astype(dtype)

        if add_bias:
            b_np = np.random.uniform(size=bias_shape).astype(dtype)
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)

        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def verify_workload_padding():
        _, _, out_height, out_width = get_const_tuple(c_np.shape)
        wkl = _get_workload(A, W, (stride, stride), padding, dilation, dtype)

        # for testing functionality,
        # we choose arbitrary int32_lanes and num_int8_elements can divide the channel,
        # regardless of the performance.
        int32_lanes, num_int8_elements = num_filter, in_channel

        # check if tile_ow candidates are the factors of the right output weight.
        cfg = autotvm.get_config()
        fallback_schedule_cpu_common_int8(cfg, wkl, int32_lanes, num_int8_elements)
        ow_tile = np.prod(cfg["tile_ow"].size)

        tvm.testing.assert_allclose(ow_tile, out_width)

    def check_target(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        if target == "cuda" and not tvm.contrib.nvcc.have_int8(dev.compute_version):
            print("Skip because int8 intrinsics are not available")
            return

        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            C = topi.cuda.conv2d_nchw_int8(
                A, W, (stride, stride), padding, (dilation, dilation), dtype
            )
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.cuda.schedule_conv2d_nchw_int8([C])

        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
        if add_bias:
            tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )
            func = tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )
            func(a, w, b, c)
        else:
            func = tvm.build(
                s,
                [A, W, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )
            func(a, w, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    verify_workload_padding()

    for target in ["cuda"]:
        check_target(target)


@pytest.mark.parametrize("in_dtype", ["int8", "uint8"])
def test_conv2d_nchw(in_dtype):
    with Int8Fallback():
        # ResNet18 workloads where channels in / out are multiple of oc_block_factor
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 56, 64, 3, 1, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 56, 64, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 56, 128, 3, 2, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 56, 128, 1, 2, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 128, 28, 128, 3, 1, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 128, 28, 256, 3, 2, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 128, 28, 256, 1, 2, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 256, 14, 256, 3, 1, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 256, 14, 512, 3, 2, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 256, 14, 512, 1, 2, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 512, 7, 512, 3, 1, 1)

        # bias, relu
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 56, 64, 3, 1, 1, add_relu=True)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 56, 64, 3, 1, 1, add_bias=True)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 56, 64, 3, 1, 1, add_bias=True, add_relu=True)

        # dilation = 2
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 56, 64, 3, 1, 1, dilation=2)

        # batch size
        verify_conv2d_NCHWc_int8(in_dtype, 4, 64, 56, 64, 3, 1, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 9, 64, 56, 64, 3, 1, 1)

        # weird workloads
        verify_conv2d_NCHWc_int8(in_dtype, 4, 4, 4, 8, 4, 4, 4)

        # inception v3 workloads where channels in / out are multiple of oc_block_factor
        verify_conv2d_NCHWc_int8(in_dtype, 1, 32, 149, 32, 3, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 32, 147, 64, 3, 1, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 73, 80, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 80, 73, 192, 3, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 192, 35, 64, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 192, 35, 48, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 48, 35, 64, 5, 1, 2)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 35, 96, 3, 1, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 96, 35, 96, 3, 1, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 192, 35, 32, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 256, 35, 64, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 256, 35, 48, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 288, 35, 64, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 288, 35, 48, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 288, 35, 384, 3, 2, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 96, 35, 96, 3, 2, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 768, 17, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 768, 17, 128, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 128, 17, 128, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 128, 17, 192, 7, 1, 3)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 128, 17, 128, 7, 1, 3)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 128, 17, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 768, 17, 160, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 160, 17, 160, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 160, 17, 192, 7, 1, 3)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 160, 17, 160, 7, 1, 3)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 160, 17, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 192, 17, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 192, 17, 192, 7, 1, 3)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 192, 17, 320, 3, 2, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 192, 17, 192, 3, 2, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 1280, 8, 320, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 1280, 8, 384, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 384, 8, 384, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 384, 8, 384, 3, 1, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 1280, 8, 448, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 448, 8, 384, 3, 1, 1)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 1280, 8, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 2048, 8, 320, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 2048, 8, 384, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 2048, 8, 448, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 2048, 8, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 1024, 19, 88, 3, 1, 1)

        # batch > 1
        verify_conv2d_NCHWc_int8(in_dtype, 7, 32, 149, 32, 3, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 8, 32, 149, 32, 3, 1, 0)
        verify_conv2d_NCHWc_int8(in_dtype, 32, 32, 149, 32, 3, 1, 0)

        # Asymmetric padding
        verify_conv2d_NCHWc_int8(in_dtype, 1, 32, 35, 64, 7, 2, (0, 0, 1, 1))
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 8, 128, 3, 1, (3, 3, 2, 2))
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 8, 64, 1, 1, (1, 2, 2, 1))
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 17, 192, 1, 1, (1, 2))
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 8, 64, 3, 1, (3, 1))
        verify_conv2d_NCHWc_int8(in_dtype, 1, 128, 8, 384, 3, 1, (0, 2))
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 8, 64, 1, 1, "VALID")
        verify_conv2d_NCHWc_int8(in_dtype, 1, 392, 8, 64, 3, 1, "VALID")
        verify_conv2d_NCHWc_int8(in_dtype, 1, 512, 19, 64, 1, 1, "SAME")
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 16, 32, 2, 1, "SAME")
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 8, 64, 3, 1, (1, 2, 2, 1), add_relu=True)
        verify_conv2d_NCHWc_int8(in_dtype, 1, 64, 8, 64, 5, 2, (1, 3), add_bias=True)
        verify_conv2d_NCHWc_int8(
            in_dtype, 1, 64, 56, 64, 3, 1, "VALID", add_bias=True, add_relu=True
        )
        verify_conv2d_NCHWc_int8(
            in_dtype, 1, 64, 56, 64, 24, 1, "SAME", add_bias=True, add_relu=True
        )

        # Conv2d NCHW int8 schedule testing. Internally, it uses NCHWc schedule. So, just
        # performing basic testing - one test for all different scenarios - batch, dilation etc..
        verify_conv2d_nchw_int8(in_dtype, 1, 64, 56, 64, 3, 1, 1)
        verify_conv2d_nchw_int8(in_dtype, 1, 64, 56, 64, 3, 1, 1, add_relu=True)
        verify_conv2d_nchw_int8(in_dtype, 1, 64, 56, 64, 3, 1, 1, dilation=2)
        verify_conv2d_nchw_int8(in_dtype, 9, 64, 56, 64, 3, 1, 1)
        verify_conv2d_nchw_int8(in_dtype, 4, 4, 4, 4, 4, 4, 4)
        verify_conv2d_nchw_int8(in_dtype, 1, 32, 149, 32, 3, 1, 0)
        verify_conv2d_nchw_int8(in_dtype, 7, 32, 149, 32, 3, 1, 0)
        verify_conv2d_nchw_int8(in_dtype, 1, 32, 35, 64, 7, 2, (0, 0, 1, 1))
        verify_conv2d_nchw_int8(in_dtype, 1, 32, 35, 64, 7, 2, (0, 0, 2, 2))


def test_conv2d_nhwc():
    with Int8Fallback():
        # Subset of inception v3 expanded (dilation > 1, batch > 1, 'VALID' padding)
        verify_conv2d_NHWC_gemm_int8(1, 3, 299, 32, 3, 2, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 32, 149, 32, 3, 1, "SAME", dilation=2)
        verify_conv2d_NHWC_gemm_int8(4, 32, 147, 64, 3, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 64, 73, 80, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 80, 73, 192, 3, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 192, 35, 48, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 192, 35, 64, 1, 1, "VALID")
        verify_conv2d_NHWC_gemm_int8(1, 192, 35, 32, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 48, 35, 64, 5, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 96, 35, 96, 3, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 256, 35, 48, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 256, 35, 64, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 288, 35, 64, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 288, 35, 48, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 96, 35, 96, 3, 2, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 128, 17, 192, 7, 1, "SAME", dilation=2)
        verify_conv2d_NHWC_gemm_int8(1, 160, 17, 160, 7, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 160, 17, 192, 1, 1, "VALID")
        verify_conv2d_NHWC_gemm_int8(1, 192, 17, 192, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 768, 5, 128, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 192, 17, 320, 3, 2, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 192, 17, 192, 3, 2, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 1280, 8, 192, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 1280, 8, 384, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 1280, 8, 320, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 1280, 8, 448, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 384, 8, 384, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 384, 8, 384, 3, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 448, 8, 384, 3, 1, "VALID")
        verify_conv2d_NHWC_gemm_int8(1, 2048, 8, 320, 1, 1, "SAME")
        verify_conv2d_NHWC_gemm_int8(1, 2048, 8, 448, 1, 1, "SAME", add_bias=True, add_relu=True)
        verify_conv2d_NHWC_gemm_int8(1, 2048, 8, 192, 1, 1, "SAME", add_bias=True)

        # Let's also verify that it compiles fine on AArch64 targets
        compile_conv2d_NHWC_gemm_int8_arm(1, 3, 299, 32, 3, 2, "SAME")


if __name__ == "__main__":
    tvm.testing.main()

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
"""Example code to do group convolution."""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple

from common import Int8Fallback
import tvm.testing


def _transform_data(data, bn):
    # NCHW -> NCHW[x]c
    batch_size, channel, height, width = data.shape
    data = np.reshape(data, (batch_size, channel // bn, bn, height, width))
    data = np.transpose(data, (0, 1, 3, 4, 2))
    return data


def _transform_kernel(kernel, ic_bn, oc_bn):
    # OIHW -> OIHW[x]o[x]i
    out_channel, in_channel, kh, kw = kernel.shape
    kernel = np.reshape(kernel, (out_channel // oc_bn, oc_bn, in_channel // ic_bn, ic_bn, kh, kw))
    kernel = np.transpose(kernel, (0, 2, 4, 5, 1, 3))
    return kernel


_group_conv2d_nchw_implement = {
    "generic": (topi.nn.group_conv2d_nchw, topi.generic.schedule_group_conv2d_nchw),
    "gpu": (topi.cuda.group_conv2d_nchw, topi.cuda.schedule_group_conv2d_nchw),
}

_group_conv2d_nhwc_implement = {
    "generic": (topi.nn.group_conv2d_nhwc, topi.generic.schedule_group_conv2d_nhwc),
}


def verify_group_conv2d_nchw(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation,
    groups,
    add_bias=False,
    add_relu=False,
):
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, groups)
    )

    in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A")
    W = te.placeholder((num_filter, in_channel // groups, kernel, kernel), name="W")
    bias = te.placeholder((num_filter, 1, 1), name="bias")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_group_conv2d.verify_group_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = tvm.topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding, groups).astype(
            dtype
        )

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
            fcompute, fschedule = tvm.topi.testing.dispatch(target, _group_conv2d_nchw_implement)
            C = fcompute(A, W, stride, padding, dilation, groups, dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = fschedule([C])

        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
        if add_bias:
            func = tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d_%d"
                % (
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    groups,
                ),
            )
            func(a, w, b, c)
        else:
            func = tvm.build(
                s,
                [A, W, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d_%d"
                % (
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    groups,
                ),
            )
            func(a, w, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    for target in ["llvm", "cuda"]:
        check_target(target)


oc_block_factor = 4
ic_block_factor = 4


def verify_group_conv2d_NCHWc_int8(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation,
    groups,
    add_bias=False,
    add_relu=False,
):
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, groups)
    )

    in_height = in_width = in_size

    A = te.placeholder(
        (batch, in_channel // ic_block_factor, in_height, in_width, ic_block_factor),
        name="A",
        dtype="int8",
    )
    W = te.placeholder(
        (
            num_filter // oc_block_factor,
            (in_channel // groups) // ic_block_factor,
            kernel,
            kernel,
            oc_block_factor,
            ic_block_factor,
        ),
        name="W",
        dtype="int8",
    )
    bias = te.placeholder(
        (num_filter // oc_block_factor, 1, 1, oc_block_factor), name="bias", dtype="int8"
    )

    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_group_conv2d.verify_group_conv2d_NCHWc_int8")
    def get_ref_data():
        a_np = np.random.randint(
            low=-128, high=127, size=(batch, in_channel, in_height, in_width)
        ).astype(dtype)
        w_np = np.random.randint(
            low=-128, high=128, size=(num_filter, in_channel // groups, kernel, kernel)
        ).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = tvm.topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding, groups).astype(
            dtype
        )

        # convert to NCHWc
        _, _, out_height, out_width = c_np.shape
        c_np = c_np.reshape(
            (batch, num_filter // oc_block_factor, oc_block_factor, out_height, out_width)
        ).transpose(0, 1, 3, 4, 2)

        if add_bias:
            b_np = np.random.uniform(size=bias_shape).astype(dtype)
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)

        return (
            _transform_data(a_np, ic_block_factor),
            _transform_kernel(w_np, ic_block_factor, oc_block_factor),
            b_np,
            c_np,
        )

    a_np, w_np, b_np, c_np = get_ref_data()

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
            C = topi.cuda.group_conv2d_NCHWc_int8(A, W, stride, padding, dilation, groups, dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.cuda.schedule_group_conv2d_NCHWc_int8([C])

        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
        if add_bias:
            func = tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d_%d"
                % (
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    groups,
                ),
            )
            func(a, w, b, c)
        else:
            func = tvm.build(
                s,
                [A, W, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d_%d"
                % (
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    groups,
                ),
            )
            func(a, w, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    for target in ["cuda"]:
        check_target(target)


def verify_group_conv2d_nchw_int8(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation,
    groups,
    add_bias=False,
    add_relu=False,
):
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, groups)
    )

    in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A", dtype="int8")
    W = te.placeholder((num_filter, in_channel // groups, kernel, kernel), name="W", dtype="int8")
    bias = te.placeholder(
        (num_filter // oc_block_factor, 1, 1, oc_block_factor), name="bias", dtype="int8"
    )

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_group_conv2d.verify_group_conv2d_nchw_int8")
    def get_ref_data():
        a_np = np.random.randint(low=-128, high=127, size=a_shape).astype(dtype)
        w_np = np.random.randint(low=-128, high=128, size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = tvm.topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding, groups).astype(
            dtype
        )

        # convert to NCHWc
        _, _, out_height, out_width = c_np.shape
        c_np = c_np.reshape(
            (batch, num_filter // oc_block_factor, oc_block_factor, out_height, out_width)
        ).transpose(0, 1, 3, 4, 2)

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
        if target == "cuda" and not tvm.contrib.nvcc.have_int8(dev.compute_version):
            print("Skip because int8 intrinsics are not available")
            return

        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            C = topi.cuda.group_conv2d_NCHWc_int8(A, W, stride, padding, dilation, groups, dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.cuda.schedule_group_conv2d_NCHWc_int8([C])

        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
        if add_bias:
            func = tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d_%d"
                % (
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    groups,
                ),
            )
            func(a, w, b, c)
        else:
            func = tvm.build(
                s,
                [A, W, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d_%d"
                % (
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    groups,
                ),
            )
            func(a, w, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    for target in ["cuda"]:
        check_target(target)


def verify_group_conv2d_nhwc(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation,
    groups,
    add_bias=False,
    add_relu=False,
):
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, groups)
    )

    in_height = in_width = in_size

    A = te.placeholder((batch, in_height, in_width, in_channel), name="A")
    W = te.placeholder((kernel, kernel, in_channel // groups, num_filter), name="W")
    bias = te.placeholder((1, 1, num_filter), name="bias")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_group_conv2d.verify_group_conv2d_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        c_np = tvm.topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding, groups).astype(
            dtype
        )

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
            fcompute, fschedule = tvm.topi.testing.dispatch(target, _group_conv2d_nhwc_implement)
            C = fcompute(A, W, stride, padding, dilation, groups, dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = fschedule([C])

        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
        if add_bias:
            func = tvm.build(
                s,
                [A, W, bias, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d_%d"
                % (
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    groups,
                ),
            )
            func(a, w, b, c)
        else:
            func = tvm.build(
                s,
                [A, W, C],
                target,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d_%d"
                % (
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    groups,
                ),
            )
            func(a, w, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    for target in ["llvm"]:
        check_target(target)


@tvm.testing.uses_gpu
def test_group_conv2d_nchw():
    # ResNeXt-50 workload
    verify_group_conv2d_nchw(1, 128, 56, 128, 3, 1, 1, 1, 32)
    verify_group_conv2d_nchw(1, 256, 56, 256, 3, 2, 1, 1, 32)
    verify_group_conv2d_nchw(1, 256, 28, 256, 3, 1, 1, 1, 32)
    verify_group_conv2d_nchw(1, 512, 28, 512, 3, 2, 1, 1, 32)
    verify_group_conv2d_nchw(1, 512, 14, 512, 3, 1, 1, 1, 32)
    verify_group_conv2d_nchw(1, 1024, 14, 1024, 3, 2, 1, 1, 32)
    verify_group_conv2d_nchw(1, 1024, 7, 1024, 3, 1, 1, 1, 32)

    # bias, relu
    verify_group_conv2d_nchw(1, 128, 56, 128, 3, 1, 1, 1, 32, add_relu=True)
    verify_group_conv2d_nchw(1, 128, 56, 128, 3, 1, 1, 1, 32, add_bias=True)
    verify_group_conv2d_nchw(1, 128, 56, 128, 3, 1, 1, 1, 32, add_relu=True, add_bias=True)

    # dilation
    verify_group_conv2d_nchw(1, 128, 56, 128, 3, 1, 1, 2, 32)

    # batch size
    verify_group_conv2d_nchw(2, 128, 56, 128, 3, 1, 1, 1, 32)
    verify_group_conv2d_nchw(9, 128, 56, 128, 3, 1, 1, 1, 32)


@tvm.testing.requires_cuda
def test_group_conv2d_NCHWc_int8():
    with Int8Fallback():
        # ResNeXt-50 workload
        verify_group_conv2d_NCHWc_int8(1, 128, 56, 128, 3, 1, 1, 1, 32)
        verify_group_conv2d_NCHWc_int8(1, 256, 56, 256, 3, 2, 1, 1, 32)
        verify_group_conv2d_NCHWc_int8(1, 256, 28, 256, 3, 1, 1, 1, 32)
        verify_group_conv2d_NCHWc_int8(1, 512, 28, 512, 3, 2, 1, 1, 32)
        verify_group_conv2d_NCHWc_int8(1, 512, 14, 512, 3, 1, 1, 1, 32)
        verify_group_conv2d_NCHWc_int8(1, 1024, 14, 1024, 3, 2, 1, 1, 32)
        verify_group_conv2d_NCHWc_int8(1, 1024, 7, 1024, 3, 1, 1, 1, 32)

        # bias, relu
        verify_group_conv2d_NCHWc_int8(1, 128, 56, 128, 3, 1, 1, 1, 32, add_relu=True)
        verify_group_conv2d_NCHWc_int8(1, 128, 56, 128, 3, 1, 1, 1, 32, add_bias=True)
        verify_group_conv2d_NCHWc_int8(
            1, 128, 56, 128, 3, 1, 1, 1, 32, add_relu=True, add_bias=True
        )
        # dilation
        verify_group_conv2d_NCHWc_int8(1, 128, 56, 128, 3, 1, 1, 2, 32)

        # batch size
        verify_group_conv2d_NCHWc_int8(2, 128, 56, 128, 3, 1, 1, 1, 32)
        verify_group_conv2d_NCHWc_int8(9, 128, 56, 128, 3, 1, 1, 1, 32)


@tvm.testing.requires_cuda
def test_group_conv2d_nchw_int8():
    with Int8Fallback():
        # ResNeXt-50 workload
        verify_group_conv2d_nchw_int8(1, 128, 56, 128, 3, 1, 1, 1, 32)
        verify_group_conv2d_nchw_int8(1, 256, 56, 256, 3, 2, 1, 1, 32)
        verify_group_conv2d_nchw_int8(1, 256, 28, 256, 3, 1, 1, 1, 32)
        verify_group_conv2d_nchw_int8(1, 512, 28, 512, 3, 2, 1, 1, 32)
        verify_group_conv2d_nchw_int8(1, 512, 14, 512, 3, 1, 1, 1, 32)
        verify_group_conv2d_nchw_int8(1, 1024, 14, 1024, 3, 2, 1, 1, 32)
        verify_group_conv2d_nchw_int8(1, 1024, 7, 1024, 3, 1, 1, 1, 32)

        # bias, relu
        verify_group_conv2d_nchw_int8(1, 128, 56, 128, 3, 1, 1, 1, 32, add_relu=True)
        verify_group_conv2d_nchw_int8(1, 128, 56, 128, 3, 1, 1, 1, 32, add_bias=True)
        verify_group_conv2d_nchw_int8(1, 128, 56, 128, 3, 1, 1, 1, 32, add_relu=True, add_bias=True)
        # dilation
        verify_group_conv2d_nchw_int8(1, 128, 56, 128, 3, 1, 1, 2, 32)

        # batch size
        verify_group_conv2d_nchw_int8(2, 128, 56, 128, 3, 1, 1, 1, 32)
        verify_group_conv2d_nchw_int8(9, 128, 56, 128, 3, 1, 1, 1, 32)


def test_group_conv2d_nhwc():
    # ResNeXt-50 workload
    verify_group_conv2d_nhwc(1, 128, 56, 128, 3, 1, 1, 1, 32)
    verify_group_conv2d_nhwc(1, 256, 56, 256, 3, 2, 1, 1, 32)
    verify_group_conv2d_nhwc(1, 256, 28, 256, 3, 1, 1, 1, 32)
    verify_group_conv2d_nhwc(1, 512, 28, 512, 3, 2, 1, 1, 32)
    verify_group_conv2d_nhwc(1, 512, 14, 512, 3, 1, 1, 1, 32)
    verify_group_conv2d_nhwc(1, 1024, 14, 1024, 3, 2, 1, 1, 32)
    verify_group_conv2d_nhwc(1, 1024, 7, 1024, 3, 1, 1, 1, 32)

    # bias, relu
    verify_group_conv2d_nhwc(1, 128, 56, 128, 3, 1, 1, 1, 32, add_relu=True)
    verify_group_conv2d_nhwc(1, 128, 56, 128, 3, 1, 1, 1, 32, add_bias=True)
    verify_group_conv2d_nhwc(1, 128, 56, 128, 3, 1, 1, 1, 32, add_relu=True, add_bias=True)

    # dilation
    verify_group_conv2d_nhwc(1, 128, 56, 128, 3, 1, 1, 2, 32)

    # batch size
    verify_group_conv2d_nhwc(2, 128, 56, 128, 3, 1, 1, 1, 32)
    verify_group_conv2d_nhwc(9, 128, 56, 128, 3, 1, 1, 1, 32)


if __name__ == "__main__":
    test_group_conv2d_nchw()
    test_group_conv2d_NCHWc_int8()
    test_group_conv2d_nchw_int8()
    test_group_conv2d_nhwc()

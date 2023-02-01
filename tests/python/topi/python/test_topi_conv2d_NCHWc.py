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
"""Test for NCHW[x]c convolution"""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import tvm.testing
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple


def _transform_data(data, bn):
    # NCHW -> NCHW[x]c
    batch_size, channel, height, width = data.shape
    data = np.reshape(data, (batch_size, channel // bn, bn, height, width))
    data = np.transpose(data, (0, 1, 3, 4, 2))
    return data


def _transform_kernel(kernel, ic_bn, oc_bn):
    # OIHW -> OIHW[x]i[x]o
    out_channel, in_channel, kh, kw = kernel.shape
    kernel = np.reshape(kernel, (out_channel // oc_bn, oc_bn, in_channel // ic_bn, ic_bn, kh, kw))
    kernel = np.transpose(kernel, (0, 2, 4, 5, 3, 1))
    return kernel


def _transform_bias(bias, bn):
    # [num_filter, 1, 1] -> [num_filter//bn, 1, 1, bn]
    num_filter, h, w = bias.shape
    bias = np.reshape(bias, (num_filter // bn, bn, h, w))
    bias = np.transpose(bias, (0, 2, 3, 1))
    return bias


def verify_conv2d_NCHWc(
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
    groups=1,
    dtype="float32",
):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    in_height = in_width = in_size
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum)
    )

    # for testing functionality,
    # we choose arbitrary block size that can divide the channel,
    # regardless of the performance.
    oc_block = 1
    for bn in range(16, 0, -1):
        if num_filter % bn == 0:
            oc_block = bn
            break

    ic_block = 1
    for bn in range(oc_block, 0, -1):
        if in_channel % bn == 0:
            ic_block = bn
            break

    A = te.placeholder((batch, in_channel // ic_block, in_height, in_width, ic_block), name="A")
    W = te.placeholder(
        (
            num_filter // oc_block,
            in_channel // ic_block // groups,
            kernel,
            kernel,
            ic_block,
            oc_block,
        ),
        name="W",
    )
    bias = te.placeholder((num_filter // oc_block, 1, 1, oc_block), name="bias")

    @memoize("topi.tests.test_topi_conv2d_NCHWc.verify_conv2d_NCHWc")
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
        w_np = np.random.uniform(size=(num_filter, in_channel // groups, kernel, kernel)).astype(
            dtype
        )
        b_np = np.random.uniform(size=(num_filter, 1, 1)).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = tvm.topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding, groups)
        if add_bias:
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)
        return (
            _transform_data(a_np, ic_block),
            _transform_kernel(w_np, ic_block, oc_block),
            _transform_bias(b_np, oc_block),
            _transform_data(c_np, oc_block),
        )

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            C = topi.x86.conv2d_NCHWc(
                A,
                W,
                (stride, stride),
                padding,
                (dilation, dilation),
                "NCHW%dc" % ic_block,
                "NCHW%dc" % oc_block,
                dtype,
            )
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.x86.schedule_conv2d_NCHWc([C])

        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
        if add_bias:
            func = tvm.build(
                s,
                [A, W, bias, C],
                device,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )
            func(a, w, b, c)
        else:
            func = tvm.build(
                s,
                [A, W, C],
                device,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
            )
            func(a, w, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    # test llvm only for now since conv2d_NCHWc implement is missing in other backend.
    for device in ["llvm"]:
        with autotvm.tophub.context(device):  # load tophub pre-tuned parameters
            check_device(device)


def test_conv2d_NCHWc():
    # ResNet18 workloads
    verify_conv2d_NCHWc(1, 3, 224, 64, 7, 2, 3)
    verify_conv2d_NCHWc(1, 64, 56, 64, 3, 1, 1)
    verify_conv2d_NCHWc(1, 64, 56, 64, 1, 1, 0)
    verify_conv2d_NCHWc(1, 64, 56, 128, 3, 2, 1)
    verify_conv2d_NCHWc(1, 64, 56, 128, 1, 2, 0)
    verify_conv2d_NCHWc(1, 128, 28, 128, 3, 1, 1)
    verify_conv2d_NCHWc(1, 128, 28, 256, 3, 2, 1)
    verify_conv2d_NCHWc(1, 128, 28, 256, 1, 2, 0)
    verify_conv2d_NCHWc(1, 256, 14, 256, 3, 1, 1)
    verify_conv2d_NCHWc(1, 256, 14, 512, 3, 2, 1)
    verify_conv2d_NCHWc(1, 256, 14, 512, 1, 2, 0)
    verify_conv2d_NCHWc(1, 512, 7, 512, 3, 1, 1)

    # bias, relu
    verify_conv2d_NCHWc(1, 64, 56, 64, 3, 1, 1, add_relu=True)
    verify_conv2d_NCHWc(1, 64, 56, 64, 3, 1, 1, add_bias=True)
    verify_conv2d_NCHWc(1, 64, 56, 64, 3, 1, 1, add_bias=True, add_relu=True)

    # dilation
    verify_conv2d_NCHWc(1, 64, 56, 64, 3, 1, 1, dilation=2)

    # batch size
    verify_conv2d_NCHWc(4, 64, 56, 64, 3, 1, 1)
    verify_conv2d_NCHWc(9, 64, 56, 64, 3, 1, 1)

    # groups
    verify_conv2d_NCHWc(1, 2048, 10, 2048, 3, 1, 1, groups=128)

    # weird workloads
    verify_conv2d_NCHWc(2, 2, 2, 2, 2, 2, 2)
    verify_conv2d_NCHWc(3, 3, 3, 3, 3, 3, 3)
    verify_conv2d_NCHWc(4, 4, 4, 4, 4, 4, 4)
    verify_conv2d_NCHWc(5, 5, 5, 5, 5, 5, 5)
    verify_conv2d_NCHWc(6, 6, 6, 6, 6, 6, 6)

    # disable these tests due to some bugs of llvm with nvptx
    # verify_conv2d_NCHWc(1, 1, 1, 1, 1, 1, 1, dilation=1)
    # verify_conv2d_NCHWc(1, 1, 1, 1, 1, 1, 1, dilation=2)
    # verify_conv2d_NCHWc(2, 13, 71, 59, 3, 1, 1)

    # inception v3 workloads
    verify_conv2d_NCHWc(1, 3, 299, 32, 3, 2, 0)
    verify_conv2d_NCHWc(1, 32, 149, 32, 3, 1, 0)
    verify_conv2d_NCHWc(1, 32, 147, 64, 3, 1, 1)
    verify_conv2d_NCHWc(1, 64, 73, 80, 1, 1, 0)
    verify_conv2d_NCHWc(1, 80, 73, 192, 3, 1, 0)
    verify_conv2d_NCHWc(1, 192, 35, 64, 1, 1, 0)
    verify_conv2d_NCHWc(1, 192, 35, 48, 1, 1, 0)
    verify_conv2d_NCHWc(1, 48, 35, 64, 5, 1, 2)
    verify_conv2d_NCHWc(1, 64, 35, 96, 3, 1, 1)
    verify_conv2d_NCHWc(1, 96, 35, 96, 3, 1, 1)
    verify_conv2d_NCHWc(1, 192, 35, 32, 1, 1, 0)
    verify_conv2d_NCHWc(1, 256, 35, 64, 1, 1, 0)
    verify_conv2d_NCHWc(1, 256, 35, 48, 1, 1, 0)
    verify_conv2d_NCHWc(1, 288, 35, 64, 1, 1, 0)
    verify_conv2d_NCHWc(1, 288, 35, 48, 1, 1, 0)
    verify_conv2d_NCHWc(1, 288, 35, 384, 3, 2, 0)
    verify_conv2d_NCHWc(1, 96, 35, 96, 3, 2, 0)
    verify_conv2d_NCHWc(1, 768, 17, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1, 768, 17, 128, 1, 1, 0)
    verify_conv2d_NCHWc(1, 128, 17, 128, 1, 1, 0)
    verify_conv2d_NCHWc(1, 128, 17, 192, 7, 1, 3)
    verify_conv2d_NCHWc(1, 128, 17, 128, 7, 1, 3)
    verify_conv2d_NCHWc(1, 128, 17, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1, 768, 17, 160, 1, 1, 0)
    verify_conv2d_NCHWc(1, 160, 17, 160, 1, 1, 0)
    verify_conv2d_NCHWc(1, 160, 17, 192, 7, 1, 3)
    verify_conv2d_NCHWc(1, 160, 17, 160, 7, 1, 3)
    verify_conv2d_NCHWc(1, 160, 17, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1, 192, 17, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1, 192, 17, 192, 7, 1, 3)
    verify_conv2d_NCHWc(1, 192, 17, 320, 3, 2, 0)
    verify_conv2d_NCHWc(1, 192, 17, 192, 3, 2, 0)
    verify_conv2d_NCHWc(1, 1280, 8, 320, 1, 1, 0)
    verify_conv2d_NCHWc(1, 1280, 8, 384, 1, 1, 0)
    verify_conv2d_NCHWc(1, 384, 8, 384, 1, 1, 0)
    verify_conv2d_NCHWc(1, 384, 8, 384, 3, 1, 1)
    verify_conv2d_NCHWc(1, 1280, 8, 448, 1, 1, 0)
    verify_conv2d_NCHWc(1, 448, 8, 384, 3, 1, 1)
    verify_conv2d_NCHWc(1, 1280, 8, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1, 2048, 8, 320, 1, 1, 0)
    verify_conv2d_NCHWc(1, 2048, 8, 384, 1, 1, 0)
    verify_conv2d_NCHWc(1, 2048, 8, 448, 1, 1, 0)
    verify_conv2d_NCHWc(1, 2048, 8, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1, 1024, 19, 84, 3, 1, 1)
    verify_conv2d_NCHWc(1, 2048, 10, 126, 3, 1, 1)
    verify_conv2d_NCHWc(1, 512, 5, 126, 3, 1, 1)
    verify_conv2d_NCHWc(1, 256, 3, 126, 3, 1, 1)

    # Asymmetric padding
    verify_conv2d_NCHWc(1, 32, 17, 64, 7, 2, (0, 0, 1, 1))
    verify_conv2d_NCHWc(1, 32, 35, 128, 3, 1, (3, 3, 2, 2))
    verify_conv2d_NCHWc(1, 32, 35, 32, 1, 1, (1, 2, 2, 1))
    verify_conv2d_NCHWc(1, 32, 17, 192, 1, 1, (1, 2))
    verify_conv2d_NCHWc(1, 32, 8, 32, 3, 1, (3, 1))
    verify_conv2d_NCHWc(1, 128, 8, 384, 3, 1, (0, 2))
    verify_conv2d_NCHWc(1, 32, 8, 32, 1, 1, "VALID")
    verify_conv2d_NCHWc(1, 388, 8, 32, 3, 1, "VALID")
    verify_conv2d_NCHWc(1, 512, 19, 32, 1, 1, "SAME")
    verify_conv2d_NCHWc(1, 32, 10, 32, 2, 1, "SAME")
    verify_conv2d_NCHWc(1, 32, 8, 32, 3, 1, (1, 2, 2, 1), add_relu=True)
    verify_conv2d_NCHWc(1, 32, 8, 32, 5, 2, (1, 3), add_bias=True)
    verify_conv2d_NCHWc(1, 32, 8, 32, 3, 1, "VALID", add_bias=True, add_relu=True)
    verify_conv2d_NCHWc(1, 32, 8, 32, 24, 1, "SAME", add_bias=True, add_relu=True)


if __name__ == "__main__":
    test_conv2d_NCHWc()

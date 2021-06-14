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

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
import tvm.testing


_conv2d_nchw_winograd_implement = {
    "arm_cpu": (topi.arm_cpu.conv2d_nchw_winograd, topi.arm_cpu.schedule_conv2d_nchw_winograd),
    "cuda": (topi.cuda.conv2d_nchw_winograd, topi.cuda.schedule_conv2d_nchw_winograd),
    "mali": (topi.mali.conv2d_nchw_winograd, topi.mali.schedule_conv2d_nchw_winograd),
}


def verify_conv2d_nchw(
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
    devices=["cuda", "llvm -device=arm_cpu", "opencl -device=mali"],
):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation)
    )

    in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A")
    W = te.placeholder((num_filter, in_channel, kernel, kernel), name="W")
    bias = te.placeholder((num_filter, 1, 1), name="bias")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_winograd.verify_conv2d_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = tvm.topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        if add_bias:
            b_np = np.random.uniform(size=bias_shape).astype(dtype)
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            fcompute, fschedule = tvm.topi.testing.dispatch(device, _conv2d_nchw_winograd_implement)
            C = fcompute(A, W, stride, padding, dilation, dtype)
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

        rtol = 1e-3
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=rtol)

    for device in devices:
        check_device(device)


@tvm.testing.uses_gpu
def test_conv2d_nchw():
    # inception v3 workloads
    verify_conv2d_nchw(1, 128, 17, 192, 7, 1, 3, devices=["cuda"])
    verify_conv2d_nchw(1, 128, 17, 128, 7, 1, 3, devices=["cuda"])
    verify_conv2d_nchw(1, 160, 17, 160, 7, 1, 3, devices=["cuda"])

    # resnet 18 workloads
    verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1)
    verify_conv2d_nchw(1, 128, 28, 128, 3, 1, 1)
    verify_conv2d_nchw(1, 256, 14, 256, 3, 1, 1)
    verify_conv2d_nchw(1, 512, 7, 512, 3, 1, 1)

    # batch size = 2
    verify_conv2d_nchw(2, 64, 56, 64, 3, 1, 1)

    # relu, bias
    verify_conv2d_nchw(2, 64, 56, 64, 3, 1, 1, add_bias=True)
    verify_conv2d_nchw(2, 64, 56, 64, 3, 1, 1, add_relu=True)
    verify_conv2d_nchw(2, 64, 56, 64, 3, 1, 1, add_relu=True, add_bias=True)

    # weird workloads
    verify_conv2d_nchw(1, 1, 1, 1, 3, 1, 1)
    verify_conv2d_nchw(3, 3, 3, 3, 3, 1, 1)
    verify_conv2d_nchw(2, 13, 71, 59, 3, 1, 1)
    verify_conv2d_nchw(1, 48, 35, 64, 5, 1, 2, devices=["cuda"])

    # Asymmetric padding
    verify_conv2d_nchw(1, 48, 56, 48, 3, 1, (1, 1, 1, 1))
    verify_conv2d_nchw(1, 64, 28, 64, 3, 1, (1, 1, 1, 1))
    verify_conv2d_nchw(1, 128, 14, 128, 3, 1, (1, 1))
    verify_conv2d_nchw(1, 512, 7, 512, 3, 1, "SAME")
    verify_conv2d_nchw(2, 13, 71, 59, 3, 1, (1, 1, 1, 1))
    verify_conv2d_nchw(2, 48, 56, 48, 3, 1, (1, 1, 1, 1), add_bias=True)
    verify_conv2d_nchw(2, 48, 56, 48, 3, 1, (1, 1), add_relu=True)
    verify_conv2d_nchw(2, 48, 56, 48, 3, 1, "SAME", add_relu=True, add_bias=True)
    verify_conv2d_nchw(1, 64, 17, 192, 7, 1, (3, 1), devices=["cuda"])
    verify_conv2d_nchw(1, 64, 17, 64, 7, 1, (3, 3, 2, 2), devices=["cuda"])
    verify_conv2d_nchw(1, 160, 17, 160, 7, 1, "SAME", devices=["cuda"])
    verify_conv2d_nchw(1, 48, 35, 48, 5, 1, "VALID", devices=["cuda"])


def verify_conv2d_nhwc(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation=1,
):
    # This version is intented to be used by the auto-scheduler,
    # so we only test the correctness of compute declaration
    # with the default naive schedule in cpu

    A = te.placeholder((batch, in_size, in_size, in_channel), name="A")
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W")
    bias = te.placeholder((1, 1, 1, num_filter), name="bias")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_winograd.verify_conv2d_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        c_np = tvm.topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    target = "llvm"
    dev = tvm.device(target)

    C = topi.nn.conv2d_winograd_nhwc(A, W, stride, padding, dilation, dtype)
    s = te.create_schedule([C.op])

    a = tvm.nd.array(a_np, device=dev)
    w = tvm.nd.array(w_np, device=dev)
    b = tvm.nd.array(b_np, device=dev)
    c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), device=dev)
    func = tvm.build(s, [A, W, C], target=target)
    func(a, w, c)

    rtol = 1e-3
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=rtol)


def test_conv2d_nhwc():
    # This version is intented to be used by the auto-scheduler,
    # so we only test the correctness of compute declaration
    # with the default naive schedule in cpu

    # resnet 18 workloads
    verify_conv2d_nhwc(1, 64, 56, 64, 3, 1, 1)
    verify_conv2d_nhwc(1, 128, 28, 128, 3, 1, 1)
    verify_conv2d_nhwc(1, 256, 14, 256, 3, 1, 1)
    verify_conv2d_nhwc(1, 512, 7, 512, 3, 1, 1)

    # more shapes
    verify_conv2d_nhwc(2, 64, 56, 64, 3, 1, 1)
    verify_conv2d_nhwc(1, 1, 1, 1, 3, 1, 1)
    verify_conv2d_nhwc(3, 3, 3, 3, 3, 1, 1)
    verify_conv2d_nhwc(2, 13, 71, 59, 3, 1, 1)

    # Asymmetric padding
    verify_conv2d_nhwc(1, 3, 7, 3, 3, 1, "SAME")
    verify_conv2d_nhwc(1, 48, 35, 48, 3, 1, "VALID")


if __name__ == "__main__":
    test_conv2d_nchw()
    test_conv2d_nhwc()

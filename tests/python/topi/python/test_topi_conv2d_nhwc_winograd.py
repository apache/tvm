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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
# pylint: disable=bad-whitespace
"""Example code to do convolution."""

import numpy as np
import tvm
from tvm import topi
import tvm.topi.testing
from tvm import te
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
import tvm.testing


_conv2d_nhwc_winograd_tensorcore = {
    "cuda": (
        topi.cuda.conv2d_nhwc_winograd_tensorcore,
        topi.cuda.schedule_conv2d_nhwc_winograd_tensorcore,
    )
}

_conv2d_nhwc_winograd_direct = {
    "cuda": (topi.cuda.conv2d_nhwc_winograd_direct, topi.cuda.schedule_conv2d_nhwc_winograd_direct)
}


def verify_conv2d_nhwc(
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
    devices="cuda",
    bgemm="direct",
):
    """Test the conv2d with winograd for nhwc layout"""
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation)
    )

    in_height = in_width = in_size

    A = te.placeholder((batch, in_height, in_width, in_channel), name="A")
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W")
    bias = te.placeholder((1, 1, 1, num_filter), name="bias")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_nhwc_winograd.verify_conv2d_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        c_np = tvm.topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        if add_bias:
            b_np = np.random.uniform(size=bias_shape).astype(dtype)
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            if bgemm == "direct":
                fcompute, fschedule = tvm.topi.testing.dispatch(
                    device, _conv2d_nhwc_winograd_direct
                )
            elif bgemm == "tensorcore":
                fcompute, fschedule = tvm.topi.testing.dispatch(
                    device, _conv2d_nhwc_winograd_tensorcore
                )
            C = fcompute(A, W, stride, padding, dilation, "float32")
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

        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=2e-3)

    check_device(devices)


@tvm.testing.requires_cuda
@tvm.testing.requires_gpu
def test_conv2d_nhwc_winograd_direct():
    """Test the conv2d with winograd for nhwc layout"""
    # resnet 18 workloads
    print("test_winograd_direct...")
    verify_conv2d_nhwc(1, 64, 56, 64, 3, 1, 1, bgemm="direct")
    verify_conv2d_nhwc(1, 128, 28, 128, 3, 1, 1)
    verify_conv2d_nhwc(1, 256, 14, 256, 3, 1, 1)
    verify_conv2d_nhwc(1, 512, 7, 512, 3, 1, 1)
    verify_conv2d_nhwc(1, 48, 35, 64, 5, 1, 2)

    # weird workloads
    verify_conv2d_nhwc(1, 1, 1, 1, 3, 1, 1)
    verify_conv2d_nhwc(3, 3, 3, 3, 3, 1, 1)
    verify_conv2d_nhwc(2, 13, 71, 59, 3, 1, 1)

    # Asymmetric padding
    verify_conv2d_nhwc(1, 512, 7, 512, 3, 1, "SAME")
    verify_conv2d_nhwc(2, 48, 56, 48, 3, 1, (1, 1), add_relu=True)
    verify_conv2d_nhwc(2, 48, 56, 48, 3, 1, "SAME", add_relu=True, add_bias=True)
    verify_conv2d_nhwc(1, 48, 35, 48, 5, 1, "VALID")


@tvm.testing.requires_cuda
@tvm.testing.requires_tensorcore
def test_conv2d_nhwc_winograd_tensorcore():
    """Test the conv2d with winograd for nhwc layout"""
    verify_conv2d_nhwc(8, 64, 56, 64, 3, 1, 1, bgemm="tensorcore")
    verify_conv2d_nhwc(8, 128, 28, 128, 3, 1, 1, bgemm="tensorcore")
    verify_conv2d_nhwc(8, 256, 14, 256, 3, 1, 1, bgemm="tensorcore")

    verify_conv2d_nhwc(2, 64, 56, 64, 3, 1, (1, 1), add_relu=True, bgemm="tensorcore")
    verify_conv2d_nhwc(2, 64, 56, 64, 3, 1, "SAME", add_relu=True, bgemm="tensorcore")


if __name__ == "__main__":
    test_conv2d_nhwc_winograd_direct()
    test_conv2d_nhwc_winograd_tensorcore()

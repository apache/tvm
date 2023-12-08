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
"""Test code for 3d convolution with winograd."""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import tvm.testing
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.nn.utils import get_pad_tuple3d
from tvm.topi.utils import get_const_tuple


_conv3d_ncdhw_implement = {
    "gpu": (topi.cuda.conv3d_ncdhw_winograd, topi.cuda.schedule_conv3d_ncdhw_winograd),
}


def verify_conv3d_ncdhw(
    batch,
    in_channel,
    in_size,
    num_filter,
    depth_kernel,
    space_kernel,
    stride,
    padding,
    dilation=1,
    add_bias=False,
    add_relu=False,
):
    pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right = get_pad_tuple3d(
        padding, (depth_kernel, space_kernel, space_kernel)
    )
    padding_sum = pad_front + pad_back + pad_top + pad_left + pad_bottom + pad_right
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, space_kernel, stride, padding_sum, dilation)
    )

    in_depth = in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_depth, in_height, in_width), name="A")
    W = te.placeholder((num_filter, in_channel, depth_kernel, space_kernel, space_kernel), name="W")
    bias = te.placeholder((num_filter, 1, 1, 1), name="bias")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv3d_ncdhw.verify_conv3d_ncdhw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation, dilation))
        c_np = tvm.topi.testing.conv3d_ncdhw_python(a_np, dw_np, stride, padding)
        if add_bias:
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
        fcompute, fschedule = tvm.topi.testing.dispatch(device, _conv3d_ncdhw_implement)
        with tvm.target.Target(device):
            C = fcompute(
                A, W, (stride, stride, stride), padding, (dilation, dilation, dilation), 1, dtype
            )
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
                % (
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    space_kernel,
                    stride,
                    padding_sum,
                    dilation,
                ),
            )
            func(a, w, b, c)
        else:
            func = tvm.build(
                s,
                [A, W, C],
                device,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (
                    batch,
                    in_channel,
                    in_size,
                    num_filter,
                    space_kernel,
                    stride,
                    padding_sum,
                    dilation,
                ),
            )
            func(a, w, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-4, atol=1e-6)

    for device in ["cuda"]:
        with autotvm.tophub.context(device):  # load tophub pre-tuned parameters
            check_device(device)


@tvm.testing.requires_gpu
def test_conv3d_ncdhw():
    # Try without depth transformation
    # 3DCNN  workloads
    verify_conv3d_ncdhw(1, 61, 20, 120, 3, 3, 1, 0)
    verify_conv3d_ncdhw(1, 61, 20, 120, 1, 3, 1, 0)
    verify_conv3d_ncdhw(1, 61, 20, 120, 5, 3, 1, 0)
    verify_conv3d_ncdhw(1, 61, 20, 120, 5, 5, 1, 2)
    verify_conv3d_ncdhw(1, 61, 20, 120, 1, 5, 1, 2)
    verify_conv3d_ncdhw(1, 61, 20, 120, 7, 7, 1, 3)
    verify_conv3d_ncdhw(1, 128, 12, 256, 3, 3, 1, 1)
    verify_conv3d_ncdhw(1, 64, 12, 128, 3, 3, 1, 1)

    # bias, relu
    verify_conv3d_ncdhw(1, 64, 12, 128, 3, 3, 1, 1, add_relu=True)
    verify_conv3d_ncdhw(1, 64, 12, 128, 3, 3, 1, 1, add_relu=True, add_bias=True)
    verify_conv3d_ncdhw(1, 64, 12, 128, 1, 3, 1, 1, add_relu=True, add_bias=True)

    # dilation = 2
    verify_conv3d_ncdhw(1, 16, 12, 16, 3, 3, 1, "VALID", dilation=2)
    verify_conv3d_ncdhw(1, 16, 12, 16, 1, 3, 1, "VALID", dilation=2)

    # batch size
    verify_conv3d_ncdhw(4, 32, 12, 64, 3, 3, 1, 1)
    verify_conv3d_ncdhw(4, 32, 12, 64, 1, 3, 1, 1)

    # weird workloads
    verify_conv3d_ncdhw(2, 2, 2, 2, 3, 3, 1, 2)
    verify_conv3d_ncdhw(3, 3, 3, 3, 3, 3, 1, 3)


if __name__ == "__main__":
    test_conv3d_ncdhw()

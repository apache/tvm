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
"""Example code to do group transpose convolution."""

import numpy as np
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import te, topi
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple

_group_conv2d_nchw_implement = {
    "generic": (
        topi.nn.group_conv2d_transpose_nchw,
        topi.generic.schedule_group_conv2d_transpose_nchw,
    ),
    "cuda": (topi.cuda.conv2d_transpose_nchw, topi.cuda.schedule_conv2d_transpose_nchw),
}


def verify_group_conv2d_transpose_nchw(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    output_padding,
    groups,
):
    print(
        "Workload: (%d, %d, %s, %d, %s, %s, %s, %s, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding, output_padding, groups)
    )

    in_height, in_width = in_size
    kernel_height, kernel_width = kernel

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A")
    W = te.placeholder((in_channel, num_filter // groups, kernel_height, kernel_width), name="W")
    bias = te.placeholder((num_filter, 1, 1), name="bias")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_group_conv2d_transpose.verify_group_conv2d_transpose_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        c_np = tvm.topi.testing.conv2d_transpose_nchw_python(
            a_np, w_np, stride, padding, output_padding, groups
        ).astype(dtype)

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
            C = fcompute(A, W, stride, padding, dtype, output_padding, groups)
            s = fschedule([C])

        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
        func = tvm.build(
            s,
            [A, W, C],
            target,
            name="group_conv2d_transpose_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d"
            % (
                batch,
                in_channel,
                in_size[0],
                in_size[1],
                num_filter,
                kernel[0],
                kernel[1],
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                padding[2],
                padding[3],
                output_padding[0],
                output_padding[1],
                groups,
            ),
        )
        func(a, w, c)
        c = c.numpy()
        for measurement, reference in zip(c, c_np):
            tvm.testing.assert_allclose(measurement, reference, rtol=1e-5)

    for target in ["llvm", "cuda"]:
        check_target(target)


@tvm.testing.uses_gpu
def test_group_conv2d_transpose_nchw():
    verify_group_conv2d_transpose_nchw(1, 4, (32, 32), 4, (5, 5), (1, 1), (0, 0, 0, 0), (0, 0), 2)
    verify_group_conv2d_transpose_nchw(1, 9, (32, 32), 9, (5, 5), (1, 1), (0, 0, 0, 0), (0, 0), 3)
    verify_group_conv2d_transpose_nchw(1, 4, (32, 32), 16, (5, 5), (2, 2), (1, 1, 1, 1), (0, 0), 4)
    verify_group_conv2d_transpose_nchw(
        1, 32, (8192, 1), 8, (31, 1), (2, 1), (14, 0, 15, 0), (0, 0), 2
    )
    verify_group_conv2d_transpose_nchw(
        1, 512, (8, 1), 256, (31, 1), (2, 1), (14, 0, 15, 0), (0, 0), 16
    )
    verify_group_conv2d_transpose_nchw(
        1, 512, (8, 1), 256, (31, 1), (2, 1), (14, 0, 15, 0), (1, 0), 16
    )
    verify_group_conv2d_transpose_nchw(
        1, 64, (64, 64), 64, (4, 4), (1, 1), (0, 0, 0, 0), (0, 0), 64
    )
    verify_group_conv2d_transpose_nchw(
        1, 128, (32, 32), 128, (4, 4), (1, 1), (0, 0, 0, 0), (0, 0), 128
    )
    verify_group_conv2d_transpose_nchw(
        1, 256, (16, 16), 256, (4, 4), (1, 1), (0, 0, 0, 0), (0, 0), 256
    )
    verify_group_conv2d_transpose_nchw(1, 1, (224, 224), 1, (1, 1), (1, 1), (0, 0, 0, 0), (0, 0), 1)
    verify_group_conv2d_transpose_nchw(
        1, 3, (224, 224), 32, (3, 3), (1, 1), (0, 0, 0, 0), (0, 0), 1
    )
    verify_group_conv2d_transpose_nchw(
        1, 3, (224, 224), 32, (3, 3), (3, 3), (0, 0, 0, 0), (0, 0), 1
    )
    verify_group_conv2d_transpose_nchw(
        1, 3, (224, 224), 32, (3, 3), (1, 1), (0, 0, 0, 0), (0, 0), 1
    )
    verify_group_conv2d_transpose_nchw(
        1, 3, (224, 224), 32, (3, 3), (2, 2), (1, 1, 1, 1), (0, 0), 1
    )
    verify_group_conv2d_transpose_nchw(1, 48, (64, 64), 12, (4, 4), (2, 2), (1, 1, 1, 1), (0, 0), 1)


if __name__ == "__main__":
    test_group_conv2d_transpose_nchw()

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
"""Test code for transposed convolution."""
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.util import get_const_tuple


_conv3d_transpose_ncdhw_implement = {
    "generic": (topi.nn.conv3d_transpose_ncdhw, topi.generic.schedule_conv3d_transpose_ncdhw),
    "cpu": (topi.x86.conv3d_transpose_ncdhw, topi.x86.schedule_conv3d_transpose_ncdhw),
    "gpu": (topi.cuda.conv3d_transpose_ncdhw, topi.cuda.schedule_conv3d_transpose_ncdhw),
}


def verify_conv3d_transpose_ncdhw(
    batch, in_channel, in_size, num_filter, kernel, stride, padding, output_padding
):
    in_depth, in_height, in_width = in_size
    kernel_depth, kernel_height, kernel_width = kernel
    stride_depth, stride_height, stride_width = stride
    pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right = padding

    A = te.placeholder((batch, in_channel, in_depth, in_height, in_width), name="A")
    W = te.placeholder(
        (in_channel, num_filter, kernel_depth, kernel_height, kernel_width), name="W"
    )

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv3d_transpose.verify_conv3d_transpose_ncdhw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = tvm.topi.testing.conv3d_transpose_ncdhw_python(
            a_np, w_np, stride, padding, output_padding
        )
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device, ctx):
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            fcompute, fschedule = tvm.topi.testing.dispatch(
                device, _conv3d_transpose_ncdhw_implement
            )
            B = fcompute(
                A,
                W,
                [stride_depth, stride_height, stride_width],
                [pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right],
                A.dtype,
                output_padding,
            )
            C = topi.nn.relu(B)
            s1 = fschedule([B])
            s2 = fschedule([C])
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)

        func1 = tvm.build(s1, [A, W, B], device)
        func2 = tvm.build(s2, [A, W, C], device)
        func1(a, w, b)
        func2(a, w, c)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, atol=1e-4, rtol=1e-4)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, atol=1e-4, rtol=1e-4)

    for device, ctx in tvm.testing.enabled_targets():
        check_device(device, ctx)


@tvm.testing.uses_gpu
def test_conv3d_transpose_ncdhw():
    verify_conv3d_transpose_ncdhw(
        1, 3, (24, 24, 24), 1, (1, 1, 1), (1, 1, 1), (0, 0, 0, 0, 0, 0), (0, 0, 0)
    )
    verify_conv3d_transpose_ncdhw(
        1, 3, (24, 24, 24), 2, (3, 3, 3), (1, 1, 1), (0, 0, 0, 0, 0, 0), (0, 0, 0)
    )
    verify_conv3d_transpose_ncdhw(
        1, 3, (24, 24, 24), 16, (3, 3, 3), (1, 1, 1), (0, 0, 0, 0, 0, 0), (0, 0, 0)
    )
    verify_conv3d_transpose_ncdhw(
        1, 3, (24, 24, 24), 16, (3, 3, 3), (3, 3, 3), (0, 0, 0, 0, 0, 0), (0, 0, 0)
    )
    verify_conv3d_transpose_ncdhw(
        1, 3, (24, 24, 24), 16, (3, 3, 3), (3, 3, 3), (0, 0, 0, 0, 0, 0), (2, 2, 2)
    )
    verify_conv3d_transpose_ncdhw(
        1, 3, (24, 24, 24), 16, (3, 3, 3), (3, 3, 3), (0, 0, 0, 0, 0, 0), (1, 0, 2)
    )
    verify_conv3d_transpose_ncdhw(
        1, 3, (24, 24, 24), 16, (3, 3, 3), (1, 1, 1), (0, 0, 0, 0, 0, 0), (0, 0, 0)
    )
    verify_conv3d_transpose_ncdhw(
        1, 3, (24, 24, 24), 16, (3, 3, 3), (2, 2, 2), (1, 1, 1, 1, 1, 1), (0, 0, 0)
    )
    verify_conv3d_transpose_ncdhw(
        1, 3, (24, 24, 24), 16, (2, 2, 2), (2, 2, 2), (0, 0, 0, 0, 0, 0), (0, 0, 0)
    )
    verify_conv3d_transpose_ncdhw(
        1, 8, (32, 32, 32), 32, (5, 5, 5), (1, 1, 1), (0, 0, 0, 0, 0, 0), (0, 0, 0)
    )
    verify_conv3d_transpose_ncdhw(
        1, 8, (32, 32, 32), 64, (5, 5, 5), (2, 2, 2), (1, 1, 1, 1, 1, 1), (0, 0, 0)
    )
    verify_conv3d_transpose_ncdhw(
        1, 8, (32, 32, 32), 64, (5, 5, 5), (2, 2, 2), (1, 1, 1, 1, 1, 1), (1, 1, 1)
    )


if __name__ == "__main__":
    test_conv3d_transpose_ncdhw()

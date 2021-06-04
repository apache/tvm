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
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple

import tvm.testing


_conv2d_transpose_nchw_implement = {
    "generic": (topi.nn.conv2d_transpose_nchw, topi.generic.schedule_conv2d_transpose_nchw),
    "cpu": (topi.x86.conv2d_transpose_nchw, topi.x86.schedule_conv2d_transpose_nchw),
    "arm_cpu": (topi.arm_cpu.conv2d_transpose_nchw, topi.arm_cpu.schedule_conv2d_transpose_nchw),
    "gpu": (topi.cuda.conv2d_transpose_nchw, topi.cuda.schedule_conv2d_transpose_nchw),
    "hls": (topi.nn.conv2d_transpose_nchw, topi.hls.schedule_conv2d_transpose_nchw),
}


def verify_conv2d_transpose_nchw(
    batch, in_channel, in_size, num_filter, kernel, stride, padding, output_padding
):
    in_height, in_width = in_size
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    pad_top, pad_left, pad_bottom, pad_right = padding

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A")
    W = te.placeholder((in_channel, num_filter, kernel_height, kernel_width), name="W")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_transpose.verify_conv2d_transpose_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = tvm.topi.testing.conv2d_transpose_nchw_python(
            a_np, w_np, stride, padding, output_padding
        )
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def check(fcompute, fschedule, target, dev):
        B = fcompute(
            A,
            W,
            [stride_height, stride_width],
            [pad_top, pad_left, pad_bottom, pad_right],
            A.dtype,
            output_padding,
        )
        C = topi.nn.relu(B)
        s1 = fschedule([B])
        s2 = fschedule([C])
        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)

        func1 = tvm.build(s1, [A, W, B], target)
        func2 = tvm.build(s2, [A, W, C], target)
        func1(a, w, b)
        func2(a, w, c)
        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    def check_generic(target, dev):
        print("Running generic on target: %s" % target)
        with tvm.target.Target(target):
            fcompute, fschedule = _conv2d_transpose_nchw_implement["generic"]
            check(fcompute, fschedule, target, dev)

    check_generic("llvm", tvm.cpu(0))

    def check_target(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            fcompute, fschedule = tvm.topi.testing.dispatch(
                target, _conv2d_transpose_nchw_implement
            )
            check(fcompute, fschedule, target, dev)

    for target, dev in tvm.testing.enabled_targets():
        check_target(target, dev)


@tvm.testing.uses_gpu
def test_conv2d_transpose_nchw():
    verify_conv2d_transpose_nchw(1, 3, (224, 224), 1, (1, 1), (1, 1), (0, 0, 0, 0), (0, 0))
    verify_conv2d_transpose_nchw(1, 3, (224, 224), 32, (3, 3), (1, 1), (0, 0, 0, 0), (0, 0))
    verify_conv2d_transpose_nchw(1, 3, (224, 224), 32, (3, 3), (3, 3), (0, 0, 0, 0), (0, 0))
    verify_conv2d_transpose_nchw(1, 3, (224, 224), 32, (3, 3), (1, 1), (0, 0, 0, 0), (0, 0))
    verify_conv2d_transpose_nchw(1, 3, (224, 224), 32, (3, 3), (2, 2), (1, 1, 1, 1), (0, 0))
    verify_conv2d_transpose_nchw(1, 3, (224, 224), 32, (3, 3), (2, 2), (1, 1, 1, 1), (1, 0))
    verify_conv2d_transpose_nchw(1, 3, (224, 224), 32, (2, 2), (2, 2), (0, 0, 0, 0), (0, 0))
    verify_conv2d_transpose_nchw(1, 3, (224, 224), 32, (2, 2), (2, 2), (0, 0, 0, 0), (1, 1))
    verify_conv2d_transpose_nchw(1, 32, (32, 32), 128, (5, 5), (1, 1), (0, 0, 0, 0), (0, 0))
    verify_conv2d_transpose_nchw(1, 32, (32, 32), 128, (5, 5), (2, 2), (1, 1, 1, 1), (0, 0))
    verify_conv2d_transpose_nchw(16, 32, (8192, 1), 8, (31, 1), (2, 1), (14, 0, 15, 0), (0, 0))
    verify_conv2d_transpose_nchw(16, 512, (8, 1), 128, (31, 1), (2, 1), (14, 0, 15, 0), (0, 0))
    verify_conv2d_transpose_nchw(16, 512, (8, 1), 128, (31, 1), (2, 1), (14, 0, 15, 0), (1, 0))


if __name__ == "__main__":
    test_conv2d_transpose_nchw()

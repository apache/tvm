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
import os
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple
import tvm.testing


_conv2d_nhwc_implement = {
    "llvm": (topi.nn.conv2d_nhwc, topi.generic.schedule_conv2d_nhwc),
    "cuda": (topi.cuda.conv2d_nhwc, topi.cuda.schedule_conv2d_nhwc),
    "cpu": (topi.nn.conv2d_nhwc, topi.x86.schedule_conv2d_nhwc),
    "arm_cpu": (
        topi.arm_cpu.conv2d_nhwc_spatial_pack,
        topi.arm_cpu.schedule_conv2d_nhwc_spatial_pack,
    ),
    "mali": (
        topi.mali.conv2d_nhwc_spatial_pack,
        topi.mali.schedule_conv2d_nhwc_spatial_pack,
    ),
    "bifrost": (
        topi.mali.conv2d_nhwc_spatial_pack,
        topi.mali.schedule_conv2d_nhwc_spatial_pack,
    ),
    "hls": (topi.nn.conv2d_nhwc, topi.hls.schedule_conv2d_nhwc),
}


def verify_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1):
    in_height = in_width = in_size

    A = te.placeholder((batch, in_height, in_width, in_channel), name="A")
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_nhwc.verify_nhwc.v2")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        b_np = tvm.topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            fcompute, fschedule = tvm.topi.testing.dispatch(target, _conv2d_nhwc_implement)
            B = fcompute(A, W, stride, padding, dilation, dtype)
            s = fschedule([B])
        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
        func = tvm.build(s, [A, W, B], target)
        func(a, w, b)
        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


@tvm.testing.uses_gpu
def test_conv2d_nhwc():
    verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, "SAME")
    verify_conv2d_nhwc(4, 128, 16, 128, 5, 2, "SAME")
    verify_conv2d_nhwc(4, 128, 16, 256, 5, 2, "SAME")
    verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, "VALID")
    verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, "VALID")
    verify_conv2d_nhwc(4, 128, 16, 128, 5, 2, "VALID")
    verify_conv2d_nhwc(4, 128, 16, 256, 5, 2, "VALID")
    verify_conv2d_nhwc(1, 128, 16, 256, 3, 2, (0, 0, 1, 1))
    verify_conv2d_nhwc(1, 128, 16, 256, 3, 2, (1, 1, 2, 2))
    verify_conv2d_nhwc(1, 128, 16, 128, 5, 2, (3, 3, 2, 2))
    verify_conv2d_nhwc(1, 128, 16, 256, 3, 2, (0, 1, 2, 3))
    # dilation = 2
    verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, "SAME", dilation=2)
    verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, (1, 1, 2, 2), dilation=2)


if __name__ == "__main__":
    test_conv2d_nhwc()

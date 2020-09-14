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
import tvm.testing
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.util import get_const_tuple


_conv3d_ndhwc_implement = {
    "generic": (topi.nn.conv3d_ndhwc, topi.generic.schedule_conv3d_ndhwc),
    "cpu": (topi.x86.conv3d_ndhwc, topi.x86.schedule_conv3d_ndhwc),
    "gpu": (topi.cuda.conv3d_ndhwc, topi.cuda.schedule_conv3d_ndhwc),
}


def verify_conv3d_ndhwc(
    batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1
):
    if isinstance(in_size, tuple):
        in_depth, in_height, in_width = in_size
    else:
        in_depth = in_height = in_width = in_size
    if isinstance(kernel, tuple):
        kernel_depth, kernel_height, kernel_width = kernel
    else:
        kernel_depth = kernel_height = kernel_width = kernel

    A = te.placeholder((batch, in_depth, in_height, in_width, in_channel), name="A")
    W = te.placeholder(
        (kernel_depth, kernel_height, kernel_width, in_channel, num_filter), name="W"
    )

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv3d_ndhwc.verify_ndhwc.v2")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, dilation, 1, 1))
        b_np = tvm.topi.testing.conv3d_ndhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()

    def check_device(device, ctx):
        print("Running on target: %s" % device)
        fcompute, fschedule = tvm.topi.testing.dispatch(device, _conv3d_ndhwc_implement)
        with tvm.target.Target(device):
            B = fcompute(A, W, stride, padding, dilation, dtype)
            s = fschedule([B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device, ctx in tvm.testing.enabled_targets():
        check_device(device, ctx)


@tvm.testing.uses_gpu
def test_conv3d_ndhwc():
    verify_conv3d_ndhwc(1, 16, 32, 16, 3, 1, "SAME")
    verify_conv3d_ndhwc(4, 32, 16, 32, 5, 2, "SAME")
    verify_conv3d_ndhwc(4, 32, 16, 64, 5, 2, "SAME")
    verify_conv3d_ndhwc(1, 64, 32, 64, 3, 1, "VALID")
    verify_conv3d_ndhwc(1, 64, 32, 64, 3, 1, "VALID")
    verify_conv3d_ndhwc(4, 32, 16, 32, 5, 2, "VALID")
    verify_conv3d_ndhwc(4, 32, 16, 64, 5, 2, "VALID")
    # dilation = 2
    verify_conv3d_ndhwc(1, 64, 32, 64, 3, 1, "SAME", dilation=2)

    verify_conv3d_ndhwc(1, 1, (20, 256, 256), 32, (1, 3, 3), (1, 2, 2), "SAME")
    verify_conv3d_ndhwc(1, 1, (20, 256, 256), 32, (1, 6, 6), (1, 2, 2), (0, 2, 2))
    verify_conv3d_ndhwc(1, 4, (20, 256, 256), 8, (1, 5, 5), (1, 2, 2), (0, 2, 2))


if __name__ == "__main__":
    test_conv3d_ndhwc()

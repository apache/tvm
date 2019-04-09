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
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple


def verify_conv2d_hwcn(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1):
    in_height = in_width = in_size

    A = tvm.placeholder((in_height, in_width, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    B = topi.nn.conv2d_hwcn(A, W, stride, padding, dilation)
    C = topi.nn.relu(B)
    s1 = topi.cuda.schedule_conv2d_hwcn([B])
    s2 = topi.cuda.schedule_conv2d_hwcn([C])

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_hwcn.verify_hwcn")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        b_np = topi.testing.conv2d_hwcn_python(a_np, dw_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np
    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        func1 = tvm.build(s1, [A, W, B], device)
        func2 = tvm.build(s2, [A, W, C], device)
        func1(a, w, b)
        func2(a, w, c)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'vulkan', 'nvptx']:
        check_device(device)


def test_conv2d_hwcn():
    verify_conv2d_hwcn(1, 256, 32, 256, 3, 1, "SAME")
    verify_conv2d_hwcn(1, 256, 32, 256, 3, 1, "SAME")
    verify_conv2d_hwcn(4, 128, 16, 128, 5, 2, "SAME")
    verify_conv2d_hwcn(4, 128, 16, 256, 5, 2, "SAME")
    verify_conv2d_hwcn(1, 256, 32, 256, 3, 1, "VALID")
    verify_conv2d_hwcn(1, 256, 32, 256, 3, 1, "VALID")
    verify_conv2d_hwcn(4, 128, 16, 128, 5, 2, "VALID")
    verify_conv2d_hwcn(4, 128, 16, 256, 5, 2, "VALID")
    # dilation = 2
    verify_conv2d_hwcn(1, 256, 32, 256, 3, 1, "SAME", dilation=2)

if __name__ == "__main__":
    test_conv2d_hwcn()

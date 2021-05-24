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


_conv2d_hwcn_implement = {
    "generic": (topi.nn.conv2d_hwcn, topi.generic.schedule_conv2d_hwcn),
    "gpu": (topi.cuda.conv2d_hwcn, topi.cuda.schedule_conv2d_hwcn),
    "opencl": (topi.cuda.conv2d_hwcn, topi.cuda.schedule_conv2d_hwcn),
}


def verify_conv2d_hwcn(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1):
    in_height = in_width = in_size

    A = te.placeholder((in_height, in_width, in_channel, batch), name="A")
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W")
    B = te.placeholder((1, num_filter, 1), name="bias")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    b_shape = get_const_tuple(B.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_hwcn.verify_hwcn")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=b_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        c1_np = tvm.topi.testing.conv2d_hwcn_python(a_np, dw_np, stride, padding)
        c2_np = c1_np + b_np
        c3_np = np.maximum(c2_np, 0)
        return a_np, w_np, b_np, c1_np, c2_np, c3_np

    a_np, w_np, b_np, c1_np, c2_np, c3_np = get_ref_data()

    def check_target(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            fcompute, fschedule = tvm.topi.testing.dispatch(target, _conv2d_hwcn_implement)
            t_conv = fcompute(A, W, stride, padding, dilation)
            t_bias = topi.add(t_conv, B)
            t_relu = topi.nn.relu(t_bias)
            s1 = fschedule([t_conv])
            s2 = fschedule([t_bias])
            s3 = fschedule([t_relu])
        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(b_np, dev)

        conv_out = tvm.nd.array(np.zeros(get_const_tuple(t_conv.shape), dtype=t_conv.dtype), dev)
        bias_out = tvm.nd.array(np.zeros(get_const_tuple(t_bias.shape), dtype=t_bias.dtype), dev)
        relu_out = tvm.nd.array(np.zeros(get_const_tuple(t_relu.shape), dtype=t_relu.dtype), dev)
        func1 = tvm.build(s1, [A, W, t_conv], target)
        func2 = tvm.build(s2, [A, W, B, t_bias], target)
        func3 = tvm.build(s3, [A, W, B, t_relu], target)
        func1(a, w, conv_out)
        func2(a, w, b, bias_out)
        func3(a, w, b, relu_out)
        tvm.testing.assert_allclose(conv_out.numpy(), c1_np, rtol=1e-5)
        tvm.testing.assert_allclose(bias_out.numpy(), c2_np, rtol=1e-5)
        tvm.testing.assert_allclose(relu_out.numpy(), c3_np, rtol=1e-5)

    for target in ["cuda", "opencl", "metal", "rocm", "vulkan", "nvptx"]:
        check_target(target)


@tvm.testing.requires_gpu
def test_conv2d_hwcn():
    verify_conv2d_hwcn(1, 256, 32, 128, 3, 1, "SAME")
    verify_conv2d_hwcn(1, 256, 32, 256, 3, 1, "SAME")
    verify_conv2d_hwcn(4, 128, 16, 128, 5, 2, "SAME")
    verify_conv2d_hwcn(4, 128, 16, 256, 5, 2, "SAME")
    verify_conv2d_hwcn(1, 256, 32, 128, 3, 1, "VALID")
    verify_conv2d_hwcn(1, 256, 32, 256, 3, 1, "VALID")
    verify_conv2d_hwcn(4, 128, 16, 128, 5, 2, "VALID")
    verify_conv2d_hwcn(4, 128, 16, 256, 5, 2, "VALID")
    # dilation = 2
    verify_conv2d_hwcn(1, 256, 32, 256, 3, 1, "SAME", dilation=2)
    # Pass stride as tuple
    verify_conv2d_hwcn(1, 256, 32, 128, 3, (1, 1), "SAME")
    verify_conv2d_hwcn(1, 256, 32, 256, 3, (1, 1), "SAME")
    verify_conv2d_hwcn(4, 128, 16, 128, 5, (2, 2), "SAME")
    verify_conv2d_hwcn(4, 128, 16, 256, 5, (2, 2), "SAME")
    verify_conv2d_hwcn(1, 256, 32, 128, 3, (1, 1), "VALID")
    verify_conv2d_hwcn(1, 256, 32, 256, 3, (1, 1), "VALID")
    verify_conv2d_hwcn(4, 128, 16, 128, 5, (2, 2), "VALID")
    verify_conv2d_hwcn(4, 128, 16, 256, 5, (2, 2), "VALID")


if __name__ == "__main__":
    test_conv2d_hwcn()

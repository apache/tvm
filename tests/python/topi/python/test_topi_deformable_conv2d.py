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
import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple

import tvm.testing


_deformable_conv2d_nchw_implement = {
    "generic": (topi.nn.deformable_conv2d_nchw, topi.generic.schedule_deformable_conv2d_nchw),
    "cuda": (topi.cuda.deformable_conv2d_nchw, topi.cuda.schedule_deformable_conv2d_nchw),
}

_deformable_conv2d_nhwc_implement = {
    "generic": (topi.nn.deformable_conv2d_nhwc, topi.generic.schedule_deformable_conv2d_nhwc),
}


def verify_deformable_conv2d_nchw(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation=1,
    deformable_groups=1,
    groups=1,
):
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d, %d, %d)"
        % (
            batch,
            in_channel,
            in_size,
            num_filter,
            kernel,
            stride,
            padding,
            dilation,
            deformable_groups,
            groups,
        )
    )

    A = te.placeholder((batch, in_channel, in_size, in_size), name="A")
    out_size = (in_size - (kernel - 1) * dilation - 1 + 2 * padding) // stride + 1
    Offset = te.placeholder(
        (batch, deformable_groups * kernel * kernel * 2, out_size, out_size), name="offset"
    )
    W = te.placeholder((num_filter, in_channel, kernel, kernel), name="W")
    bias = te.placeholder((num_filter, 1, 1), name="bias")

    a_shape = get_const_tuple(A.shape)
    offset_shape = get_const_tuple(Offset.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_deformable_conv2d_nchw.verify_deformable_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        offset_np = np.random.randn(*offset_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        c_np = tvm.topi.testing.deformable_conv2d_nchw_python(
            a_np, offset_np, w_np, stride, padding, dilation, deformable_groups, groups
        )

        return a_np, offset_np, w_np, c_np

    a_np, offset_np, w_np, c_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        fcompute, fschedule = tvm.topi.testing.dispatch(device, _deformable_conv2d_nchw_implement)
        with tvm.target.Target(device):
            C = fcompute(A, Offset, W, stride, padding, dilation, deformable_groups, groups, dtype)
            s = fschedule([C])

            a = tvm.nd.array(a_np, dev)
            offset = tvm.nd.array(offset_np, dev)
            w = tvm.nd.array(w_np, dev)
            c = tvm.nd.empty(c_np.shape, dtype=c_np.dtype, device=dev)

            func = tvm.build(s, [A, Offset, W, C], device)
            func(a, offset, w, c)
            tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    for device in ["llvm", "cuda"]:
        check_device(device)


def verify_deformable_conv2d_nhwc(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation=1,
    deformable_groups=1,
    groups=1,
):
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d, %d, %d)"
        % (
            batch,
            in_channel,
            in_size,
            num_filter,
            kernel,
            stride,
            padding,
            dilation,
            deformable_groups,
            groups,
        )
    )

    A = te.placeholder((batch, in_size, in_size, in_channel), name="A")
    out_size = (in_size - (kernel - 1) * dilation - 1 + 2 * padding) // stride + 1
    Offset = te.placeholder(
        (batch, out_size, out_size, deformable_groups * kernel * kernel * 2), name="offset"
    )
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W")
    bias = te.placeholder((num_filter,), name="bias")

    a_shape = get_const_tuple(A.shape)
    offset_shape = get_const_tuple(Offset.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_deformable_conv2d_nchw.verify_deformable_conv2d_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        offset_np = np.random.randn(*offset_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        c_np = tvm.topi.testing.deformable_conv2d_nhwc_python(
            a_np, offset_np, w_np, stride, padding, dilation, deformable_groups, groups
        )

        return a_np, offset_np, w_np, c_np

    a_np, offset_np, w_np, c_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        fcompute, fschedule = tvm.topi.testing.dispatch(device, _deformable_conv2d_nhwc_implement)
        with tvm.target.Target(device):
            C = fcompute(A, Offset, W, stride, padding, dilation, deformable_groups, groups, dtype)
            s = fschedule([C])

            a = tvm.nd.array(a_np, dev)
            offset = tvm.nd.array(offset_np, dev)
            w = tvm.nd.array(w_np, dev)
            c = tvm.nd.empty(c_np.shape, dtype=c_np.dtype, device=dev)

            func = tvm.build(s, [A, Offset, W, C], device)
            func(a, offset, w, c)
            tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    for device in ["llvm"]:
        check_device(device)


@tvm.testing.uses_gpu
def test_deformable_conv2d_nchw():
    verify_deformable_conv2d_nchw(1, 16, 7, 16, 1, 1, 0, deformable_groups=4)
    verify_deformable_conv2d_nchw(1, 16, 7, 16, 3, 1, 1, dilation=2, deformable_groups=4)
    verify_deformable_conv2d_nchw(1, 16, 7, 16, 3, 1, 2, dilation=2)


def test_deformable_conv2d_nhwc():
    verify_deformable_conv2d_nhwc(1, 16, 7, 16, 1, 1, 0, deformable_groups=4)
    verify_deformable_conv2d_nhwc(1, 16, 7, 16, 3, 1, 1, dilation=2, deformable_groups=4)
    verify_deformable_conv2d_nhwc(1, 16, 7, 16, 3, 1, 2, dilation=2)


if __name__ == "__main__":
    test_deformable_conv2d_nchw()
    test_deformable_conv2d_nhwc()

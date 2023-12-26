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
import platform
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple
import tvm.testing


_conv2d_nhwc_implement = {
    "generic": (topi.nn.conv2d_nhwc, topi.generic.schedule_conv2d_nhwc),
    "gpu": (topi.gpu.conv2d_nhwc, topi.gpu.schedule_conv2d_nhwc),
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

device = tvm.testing.parameter(
    (
        "llvm --device arm_cpu --mtriple aarch64-linux-gnu",
        topi.arm_cpu.conv2d_nhwc_spatial_pack,
        topi.arm_cpu.schedule_conv2d_nhwc_spatial_pack,
    ),
    (
        "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v8.2a",
        topi.arm_cpu.compute_conv2d_NHWC_hybrid,
        topi.arm_cpu.schedule_conv2d_NHWC_hybrid,
    ),
)

dtype = tvm.testing.parameter("float32")

batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation = tvm.testing.parameters(
    (1, 256, 32, 256, 3, 1, "SAME", 1),
    (4, 128, 16, 128, 5, 2, "SAME", 1),
    (4, 128, 16, 256, 5, 2, "SAME", 1),
    (1, 256, 32, 256, 3, 1, "VALID", 1),
    (1, 256, 32, 256, 3, 1, "VALID", 1),
    (4, 128, 16, 128, 5, 2, "VALID", 1),
    (4, 128, 16, 256, 5, 2, "VALID", 1),
    (1, 128, 16, 256, 3, 2, (0, 0, 1, 1), 1),
    (1, 128, 16, 256, 3, 2, (1, 1, 2, 2), 1),
    (1, 128, 16, 128, 5, 2, (3, 3, 2, 2), 1),
    (1, 128, 16, 256, 3, 2, (0, 1, 2, 3), 1),
    (1, 256, 32, 256, 3, 1, "SAME", 2),
    (1, 256, 32, 256, 3, 1, (1, 1, 2, 2), 2),
)


@tvm.testing.fixture(cache_return_value=True)
def ref_data(dtype, batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation):
    in_height = in_width = in_size
    a_shape = (batch, in_height, in_width, in_channel)
    w_shape = (kernel, kernel, in_channel, num_filter)

    a_np = np.random.uniform(size=a_shape).astype(dtype)
    w_np = np.random.uniform(size=w_shape).astype(dtype)
    dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
    b_np = tvm.topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
    return a_np, w_np, b_np


def test_conv2d_nhwc_gemm_fp32(device, ref_data, dtype, stride, padding, dilation):
    a_np, w_np, b_np = ref_data

    A = te.placeholder(a_np.shape, name="A", dtype=dtype)
    W = te.placeholder(w_np.shape, name="W", dtype=dtype)

    target, compute, schedule = device
    dev = tvm.device(target, 0)

    with tvm.target.Target(target):
        B = compute(A, W, stride, padding, dilation, dtype)
        s = schedule([B])
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    func = tvm.build(s, [A, W, B], target)

    build_only = platform.machine() != "aarch64"
    if build_only:
        return

    func(a, w, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


def test_conv2d_nhwc_hwio(target, dev, ref_data, dtype, stride, padding, dilation):
    a_np, w_np, b_np = ref_data

    A = te.placeholder(a_np.shape, name="A", dtype=dtype)
    W = te.placeholder(w_np.shape, name="W", dtype=dtype)

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


def test_conv2d_nhwc_ohwi(ref_data, dtype, stride, padding, dilation):
    # only test on CPU target because topi doesn't have schedules for this layout
    target = "llvm"
    dev = tvm.device(target, 0)
    a_np, w_np_hwio, b_np = ref_data
    w_np_ohwi = w_np_hwio.transpose(3, 0, 1, 2)  # HWIO -> OHWI

    A = te.placeholder(a_np.shape, name="A", dtype=dtype)
    W = te.placeholder(w_np_ohwi.shape, name="W", dtype=dtype)

    B = topi.nn.conv2d(
        A,
        W,
        stride,
        padding,
        dilation,
        data_layout="NHWC",
        kernel_layout="OHWI",
        out_dtype="float32",
    )
    s = tvm.te.create_schedule(B.op)
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np_ohwi, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    func = tvm.build(s, [A, W, B], target)
    func(a, w, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()

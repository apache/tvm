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
import pytest
import numpy as np
import tvm
from tvm import te
from tvm import topi
from tvm.target.codegen import llvm_version_major
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
        False,
    ),
    (
        "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v8.2a,+fullfp16",
        topi.arm_cpu.compute_conv2d_NHWC_hybrid,
        topi.arm_cpu.schedule_conv2d_NHWC_hybrid,
        False,
    ),
    (
        "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v8.6a,+sve",
        topi.arm_cpu.compute_conv2d_NHWC_hybrid_SVE,
        topi.arm_cpu.schedule_conv2d_NHWC_hybrid_SVE,
        False,
    ),
    (
        "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v8.2a,+fullfp16",
        topi.arm_cpu.compute_conv2d_NHWC_hybrid,
        topi.arm_cpu.schedule_conv2d_NHWC_hybrid_TIR,
        True,
    ),
    (
        "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v8.6a,+sve",
        topi.arm_cpu.compute_conv2d_NHWC_hybrid_SVE,
        topi.arm_cpu.schedule_conv2d_NHWC_hybrid_TIR,
        True,
    ),
    (
        "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v9a,+sme",
        topi.arm_cpu.compute_conv2d_NHWC_hybrid_SME,
        topi.arm_cpu.schedule_conv2d_NHWC_hybrid_TIR,
        True,
    ),
)

dtype = tvm.testing.parameter("float16", "float32")

batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation = tvm.testing.parameters(
    # Pad M, N, K
    (1, 1, 1, 1, 1, 1, "SAME", 1),
    (1, 1, 3, 15, 1, 1, "SAME", 1),
    # Pad M, K
    (1, 3, 9, 16, 3, 1, "SAME", 1),
    # Pad M, N
    (1, 2, 9, 15, 4, 1, "SAME", 1),
    # Pad K, N
    (1, 7, 4, 15, 3, 1, "SAME", 1),
    # Pad M
    (1, 2, 9, 16, 4, 1, "SAME", 1),
    # Pad K
    (1, 7, 4, 16, 3, 1, "SAME", 1),
    # Pad N
    (1, 2, 4, 15, 4, 1, "SAME", 1),
    (1, 2, 4, 20, 1, 1, "SAME", 1),
    # Large workloads
    (1, 256, 32, 256, 3, 1, "SAME", 1),
    (4, 128, 16, 128, 5, 2, "SAME", 1),
    (4, 128, 16, 256, 5, 2, "SAME", 1),
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

    np.random.seed(0)
    a_np = np.random.uniform(size=a_shape).astype(dtype)
    w_np = np.random.uniform(size=w_shape).astype(dtype)
    dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))

    # scipy.signal.convolve2d does not support float16 data types,
    # and the python fallback would be too slow for general use.
    conv_dtype = "float32" if dtype == "float16" else dtype
    b_np = tvm.topi.testing.conv2d_nhwc_python(
        a_np.astype(conv_dtype), dw_np.astype(conv_dtype), stride, padding
    ).astype(dtype)
    return a_np, w_np, b_np


def get_tolerance(dtype, w_np, b_np):
    if dtype == "float16":
        # A summation in float16 with a single accumulator very
        # quickly runs into large rounding errors.
        # This tolerance is necessary to ensure no false negatives,
        # but it may introduce false positives, depending on schedule behaviour.
        num_values_summed = w_np.shape[0] * w_np.shape[1] * w_np.shape[2]
        next_float_gap_size = np.nextafter(b_np.max(), np.inf, dtype=b_np.dtype) - b_np.max()
        tol = {"rtol": 1e-5, "atol": num_values_summed * next_float_gap_size / 2}
    else:
        tol = {"rtol": 1e-5, "atol": 1e-7}

    return tol


def test_conv2d_nhwc_gemm(device, ref_data, dtype, stride, padding, dilation):
    a_np, w_np, b_np = ref_data

    A = te.placeholder(a_np.shape, name="A", dtype=dtype)
    W = te.placeholder(w_np.shape, name="W", dtype=dtype)

    target_string, compute, schedule, use_tir_schedule = device
    dev = tvm.device(target_string, 0)
    target = tvm.target.Target(target_string)

    if target.features.has_sve and llvm_version_major() < 15:
        pytest.skip(f"LLVM {llvm_version_major()} does not support targeting SVE.")

    if target.features.has_sme and llvm_version_major() < 16:
        pytest.skip(f"LLVM {llvm_version_major()} does not support targeting SME.")

    if target.features.has_sme and a_np.shape[0] > 1:
        pytest.skip(f"Conv2d with batches > 1 targeting SME not implemented.")

    if target.features.has_sme and (a_np.shape[3] * w_np.shape[0] * w_np.shape[1]) <= 1:
        pytest.skip(f"Conv2d with unit reduction dimension targeting SME not supported.")

    # SME schedule always outputs float32 results, regardless of input dtype.
    # Otherwise, output dtype is the same as input dtype.
    out_dtype = "float32" if target.features.has_sme else dtype

    with target:
        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        B = compute(A, W, stride, padding, dilation, out_dtype)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
        if use_tir_schedule:
            primfunc = te.create_prim_func([A, W, B])
            sch = schedule(tvm.tir.Schedule(primfunc))
            func = tvm.build(sch.mod["main"], target)
        else:
            s = schedule([B])
            func = tvm.build(s, [A, W, B], target)

        # Run only on AArch64 devices
        # Do not run SVE/SME schedules on non-SVE/SME devices
        build_only = (
            platform.machine() != "aarch64"
            or (
                dtype == "float16"
                and target.features.has_fp16_simd
                and not tvm.testing.requires_arm_fp16.run_time_check()
            )
            or (target.features.has_sve and not tvm.testing.requires_aarch64_sve.run_time_check())
            or (target.features.has_sme and not tvm.testing.requires_aarch64_sme.run_time_check())
        )
        if build_only:
            return

        func(a, w, b)
    tol = get_tolerance(out_dtype, w_np, b_np)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=tol["rtol"], atol=tol["atol"])


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
    tol = get_tolerance(dtype, w_np, b_np)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=tol["rtol"], atol=tol["atol"])


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
    tol = get_tolerance(dtype, w_np_hwio, b_np)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=tol["rtol"], atol=tol["atol"])


if __name__ == "__main__":
    tvm.testing.main()

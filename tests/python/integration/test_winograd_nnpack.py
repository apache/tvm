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
"""Test winograd convolution using nnpack impl."""
import numpy as np
from pytest import skip

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import autotvm, te, topi
from tvm.autotvm.task.space import FallbackConfigEntity
from tvm.contrib import nnpack
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple


def verify_conv2d_nchw(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    devices,
    dilation=1,
    add_bias=False,
    add_relu=False,
):
    """Verify conv2d nchw workload."""
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation)
    )

    in_height = in_width = in_size

    placholder_a = te.placeholder((batch, in_channel, in_height, in_width), name="A")
    placeholder_w = te.placeholder((num_filter, in_channel, kernel, kernel), name="W")
    bias = te.placeholder((num_filter, 1, 1), name="bias")

    a_shape = get_const_tuple(placholder_a.shape)
    w_shape = get_const_tuple(placeholder_w.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = placholder_a.dtype

    @memoize("topi.tests.test_topi_conv2d_nchw.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = tvm.topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        if add_bias:
            b_np = np.random.uniform(size=bias_shape).astype(dtype)
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skipping %s becuase it is not enabled" % device)
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            result_c = topi.nn.conv2d(
                placholder_a,
                placeholder_w,
                stride,
                padding,
                dilation,
                data_layout="NCHW",
                out_dtype=dtype,
            )
            if add_bias:
                result_c = topi.add(result_c, bias)
            if add_relu:
                result_c = topi.nn.relu(result_c)
            schedule = topi.generic.schedule_conv2d_nchw([result_c])

        buff_a = tvm.nd.array(a_np, dev)
        buff_w = tvm.nd.array(w_np, dev)
        buff_b = tvm.nd.array(b_np, dev)
        buff_c = tvm.nd.array(np.zeros(get_const_tuple(result_c.shape), dtype=result_c.dtype), dev)
        if add_bias:
            func = tvm.build(
                schedule,
                [placholder_a, placeholder_w, bias, result_c],
                device,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation),
            )
            func(buff_a, buff_w, buff_b, buff_c)
        else:
            func = tvm.build(
                schedule,
                [placholder_a, placeholder_w, result_c],
                device,
                name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
                % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation),
            )
            func(buff_a, buff_w, buff_c)
        tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-4)

    for device in devices:
        check_device(device)


class WinogradFallback(autotvm.FallbackContext):
    """Winograd fallbacks."""

    def _query_inside(self, target, workload):
        key = (target, workload)
        if key in self.memory:
            return self.memory[key]
        cfg = FallbackConfigEntity()
        cfg.template_key = "winograd_nnpack_fp32"
        self.memory[key] = cfg
        return cfg


def test_conv2d_nchw():
    """Verify conv2d nchw winograd works."""

    if not tvm.get_global_func(
        "tvm.contrib.nnpack.convolution_inference_without_weight_transform", True
    ):
        skip("extern function is not available")

    if not nnpack.is_available():
        skip("nnpack is not available")

    devices = ["llvm -device=arm_cpu"]
    autotvm.GLOBAL_SCOPE.silent = True
    with WinogradFallback():
        # resnet 18 workloads
        verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1, devices=devices)
        verify_conv2d_nchw(1, 128, 28, 128, 3, 1, 1, devices=devices)
        verify_conv2d_nchw(1, 256, 14, 256, 3, 1, 1, devices=devices)
        verify_conv2d_nchw(1, 512, 7, 512, 3, 1, 1, devices=devices)

        # unet workloads
        verify_conv2d_nchw(1, 3, 192, 12, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 4, 192, 12, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 12, 96, 24, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 24, 48, 48, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 48, 24, 96, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 96, 12, 180, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 180, 6, 220, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 220, 6, 180, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 180, 12, 96, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 96, 24, 48, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 48, 48, 24, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 24, 96, 12, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 12, 192, 1, 3, 1, 1, add_bias=True, devices=devices)

        # relu, bias
        verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1, add_bias=True, devices=devices)
        verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1, add_relu=True, devices=devices)
        verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1, add_relu=True, add_bias=True, devices=devices)

        # werid workloads
        verify_conv2d_nchw(1, 3, 3, 3, 3, 1, 1, devices=devices)
        verify_conv2d_nchw(1, 13, 71, 59, 3, 1, 1, devices=devices)
    autotvm.GLOBAL_SCOPE.silent = False


if __name__ == "__main__":
    tvm.testing.main()

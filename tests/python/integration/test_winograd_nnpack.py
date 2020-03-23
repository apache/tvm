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
from tvm.autotvm.task.space import FallbackConfigEntity
from tvm.contrib import nnpack
from tvm.contrib.pickle_memoize import memoize
import topi
import topi.testing
from topi.util import get_const_tuple
from pytest import skip


def verify_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1, add_bias=False, add_relu=False,
        devices=['cuda', 'llvm -device=arm_cpu', 'opencl -device=mali']):
    print("Workload: (%d, %d, %d, %d, %d, %d, %d, %d)" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))

    in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = te.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    bias = te.placeholder((num_filter, 1, 1), name='bias')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_nchw.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        if add_bias:
            b_np = np.random.uniform(size=bias_shape).astype(dtype)
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            skip("s is not enabled" % device)
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            C = topi.nn.conv2d(A, W, stride, padding, dilation, layout='NCHW', out_dtype=dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.generic.schedule_conv2d_nchw([C])

        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        if add_bias:
            func = tvm.build(s, [A, W, bias, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func(a, w, b, c)
        else:
            func = tvm.build(s, [A, W, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func(a, w, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-4)


    for device in devices:
        check_device(device)


class WinogradFallback(autotvm.FallbackContext):
    def _query_inside(self, target, workload):
        key = (target, workload)
        if key in self.memory:
            return self.memory[key]
        cfg = FallbackConfigEntity()
        cfg.template_key = 'winograd_nnpack_fp32'
        self.memory[key] = cfg
        return cfg

def test_conv2d_nchw():
    if not tvm.get_global_func("tvm.contrib.nnpack.convolution_inference_without_weight_transform", True):
        skip("extern function is not available")

    if not nnpack.is_available():
        skip("nnpack is not available")

    devices = ['llvm -device=arm_cpu']
    autotvm.DispatchContext.current.silent = True
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


if __name__ == "__main__":
    import pytest
    pytest.main()

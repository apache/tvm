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

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.nn.util import get_pad_tuple
from topi.util import get_const_tuple
from topi.arm_cpu.conv2d_gemm import is_aarch64_arm

from common import get_all_backend, Int8Fallback

def compile_conv2d_NHWC_gemm_int8_arm(batch, in_channel, in_size, num_filter, kernel, stride, padding,
                                 dilation=1, add_bias=False, add_relu=False):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print("Workload: (%d, %d, %d, %d, %d, %d, %d, %d)" % (batch, in_channel, in_size, num_filter,
                                                          kernel, stride, padding_sum, dilation))

    in_height = in_width = in_size
    A = te.placeholder((batch, in_height, in_width, in_channel), name='A', dtype='int8')
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name='W', dtype='int8')
    bias = te.placeholder((num_filter,), name='bias', dtype='int8')
    dtype = 'int32'
    device = "llvm --device arm_cpu --mtriple aarch64-linux-gnu"

    ctx = tvm.context(device, 0)
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return
    print("Compiling on arm AArch64 target: %s" % device)
    with tvm.target.create(device):
        assert is_aarch64_arm(), "AArch64 target not recognized"

        C = topi.arm_cpu.compute_conv2d_NHWC_quantized(A, W, (stride, stride), padding,
                                                       (dilation, dilation), dtype)
        if add_bias:
            C = topi.add(C, bias)
        if add_relu:
            C = topi.nn.relu(C)
        s = topi.arm_cpu.schedule_conv2d_NHWC_quantized([C])

    if add_bias:
        tvm.build(s, [A, W, bias, C], device,
                  name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch,
                                                         in_channel,
                                                         in_size,
                                                         num_filter,
                                                         kernel,
                                                         stride,
                                                         padding_sum,
                                                         dilation))
        func = tvm.build(s, [A, W, bias, C], device,
                         name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch,
                                                                in_channel,
                                                                in_size,
                                                                num_filter,
                                                                kernel,
                                                                stride,
                                                                padding_sum,
                                                                dilation))
    else:
        func = tvm.build(s, [A, W, C], device,
                         name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch,
                                                                in_channel,
                                                                in_size,
                                                                num_filter,
                                                                kernel,
                                                                stride,
                                                                padding_sum,
                                                                dilation))

def verify_conv2d_NHWC_gemm_int8(batch, in_channel, in_size, num_filter, kernel, stride, padding,
                                 dilation=1, add_bias=False, add_relu=False):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print("Workload: (%d, %d, %d, %d, %d, %d, %d, %d)" % (batch, in_channel, in_size, num_filter,
                                                          kernel, stride, padding_sum, dilation))

    in_height = in_width = in_size

    A = te.placeholder((batch, in_height, in_width, in_channel), name='A', dtype='int8')
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name='W', dtype='int8')
    bias = te.placeholder((num_filter,), name='bias', dtype='int8')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_int8.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.randint(low=-128, high=127, size=a_shape).astype(dtype)
        w_np = np.random.randint(low=-128, high=128, size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        c_np = topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding).astype(dtype)

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
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            C = topi.arm_cpu.compute_conv2d_NHWC_quantized(A, W, (stride, stride), padding,
                                                           (dilation, dilation), dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.arm_cpu.schedule_conv2d_NHWC_quantized([C])

        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        if add_bias:
            tvm.build(s, [A, W, bias, C], device,
                      name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch,
                                                             in_channel,
                                                             in_size,
                                                             num_filter,
                                                             kernel,
                                                             stride,
                                                             padding_sum,
                                                             dilation))
            func = tvm.build(s, [A, W, bias, C], device,
                             name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch,
                                                                    in_channel,
                                                                    in_size,
                                                                    num_filter,
                                                                    kernel,
                                                                    stride,
                                                                    padding_sum,
                                                                    dilation))
            func(a, w, b, c)
        else:
            func = tvm.build(s, [A, W, C], device,
                             name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch,
                                                                    in_channel,
                                                                    in_size,
                                                                    num_filter,
                                                                    kernel,
                                                                    stride,
                                                                    padding_sum,
                                                                    dilation))
            func(a, w, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    check_device("llvm")

oc_block_factor = 4
def verify_conv2d_NCHWc_int8(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1, add_bias=False, add_relu=False):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print("Workload: (%d, %d, %d, %d, %d, %d, %d, %d)" % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))

    in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_height, in_width), name='A', dtype='int8')
    W = te.placeholder((num_filter, in_channel, kernel, kernel), name='W', dtype='int8')
    bias = te.placeholder((num_filter // oc_block_factor, 1, 1, oc_block_factor), name='bias',
                            dtype='int8')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_int8.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.randint(low=-128, high=127, size=a_shape).astype(dtype)
        w_np = np.random.randint(low=-128, high=128, size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding).astype(dtype)

        # convert to NCHWc
        _, _, out_height, out_width = c_np.shape
        c_np = c_np.reshape((batch, num_filter // oc_block_factor, oc_block_factor, \
                out_height, out_width)).transpose(0, 1, 3, 4, 2)

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
            print("Skip because %s is not enabled" % device)
            return
        if device == "cuda" and not tvm.contrib.nvcc.have_int8(ctx.compute_version):
            print("Skip because int8 intrinsics are not available")
            return

        print("Running on target: %s" % device)
        with tvm.target.create(device):
            C = topi.cuda.conv2d_NCHWc_int8(A, W, (stride, stride), padding, (dilation, dilation),
                                            'NCHW', dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.cuda.schedule_conv2d_NCHWc_int8([C])

        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        if add_bias:
            tvm.build(s, [A, W, bias, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))
            func = tvm.build(s, [A, W, bias, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))
            func(a, w, b, c)
        else:
            func = tvm.build(s, [A, W, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))
            func(a, w, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in ["cuda"]:
        check_device(device)


def verify_conv2d_nchw_int8(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1, add_bias=False, add_relu=False):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print("Workload: (%d, %d, %d, %d, %d, %d, %d, %d)" % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))

    in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_height, in_width), name='A', dtype='int8')
    W = te.placeholder((num_filter, in_channel, kernel, kernel), name='W', dtype='int8')
    bias = te.placeholder((num_filter, 1, 1), name='bias', dtype='int8')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_int8.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.randint(low=-128, high=127, size=a_shape).astype(dtype)
        w_np = np.random.randint(low=-128, high=128, size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding).astype(dtype)

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
            print("Skip because %s is not enabled" % device)
            return
        if device == "cuda" and not tvm.contrib.nvcc.have_int8(ctx.compute_version):
            print("Skip because int8 intrinsics are not available")
            return

        print("Running on target: %s" % device)
        with tvm.target.create(device):
            C = topi.cuda.conv2d_nchw_int8(A, W, (stride, stride), padding, (dilation, dilation),
                                           dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.cuda.schedule_conv2d_nchw_int8([C])

        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        if add_bias:
            tvm.build(s, [A, W, bias, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))
            func = tvm.build(s, [A, W, bias, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))
            func(a, w, b, c)
        else:
            func = tvm.build(s, [A, W, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))
            func(a, w, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in ["cuda"]:
        check_device(device)


def test_conv2d_nchw():
    with Int8Fallback():
        # ResNet18 workloads where channels in / out are multiple of oc_block_factor
        verify_conv2d_NCHWc_int8(1,  64,  56,  64, 3, 1, 1)
        verify_conv2d_NCHWc_int8(1,  64,  56,  64, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  64,  56, 128, 3, 2, 1)
        verify_conv2d_NCHWc_int8(1,  64,  56, 128, 1, 2, 0)
        verify_conv2d_NCHWc_int8(1, 128,  28, 128, 3, 1, 1)
        verify_conv2d_NCHWc_int8(1, 128,  28, 256, 3, 2, 1)
        verify_conv2d_NCHWc_int8(1, 128,  28, 256, 1, 2, 0)
        verify_conv2d_NCHWc_int8(1, 256,  14, 256, 3, 1, 1)
        verify_conv2d_NCHWc_int8(1, 256,  14, 512, 3, 2, 1)
        verify_conv2d_NCHWc_int8(1, 256,  14, 512, 1, 2, 0)
        verify_conv2d_NCHWc_int8(1, 512,   7, 512, 3, 1, 1)

        # bias, relu
        verify_conv2d_NCHWc_int8(1, 64, 56, 64, 3, 1, 1, add_relu=True)
        verify_conv2d_NCHWc_int8(1, 64, 56, 64, 3, 1, 1, add_bias=True)
        verify_conv2d_NCHWc_int8(1, 64, 56, 64, 3, 1, 1, add_bias=True, add_relu=True)

        # dilation = 2
        verify_conv2d_NCHWc_int8(1, 64, 56, 64, 3, 1, 1, dilation=2)

        # batch size
        verify_conv2d_NCHWc_int8(4, 64, 56, 64, 3, 1, 1)
        verify_conv2d_NCHWc_int8(9, 64, 56, 64, 3, 1, 1)

        # weird workloads
        verify_conv2d_NCHWc_int8(4, 4, 4, 4, 4, 4, 4)

        # inception v3 workloads where channels in / out are multiple of oc_block_factor
        verify_conv2d_NCHWc_int8(1,   32, 149,  32, 3, 1, 0)
        verify_conv2d_NCHWc_int8(1,   32, 147,  64, 3, 1, 1)
        verify_conv2d_NCHWc_int8(1,   64,  73,  80, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,   80,  73, 192, 3, 1, 0)
        verify_conv2d_NCHWc_int8(1,  192,  35,  64, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  192,  35,  48, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,   48,  35,  64, 5, 1, 2)
        verify_conv2d_NCHWc_int8(1,   64,  35,  96, 3, 1, 1)
        verify_conv2d_NCHWc_int8(1,   96,  35,  96, 3, 1, 1)
        verify_conv2d_NCHWc_int8(1,  192,  35,  32, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  256,  35,  64, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  256,  35,  48, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  288,  35,  64, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  288,  35,  48, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  288,  35, 384, 3, 2, 0)
        verify_conv2d_NCHWc_int8(1,   96,  35,  96, 3, 2, 0)
        verify_conv2d_NCHWc_int8(1,  768,  17, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  768,  17, 128, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  128,  17, 128, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  128,  17, 192, 7, 1, 3)
        verify_conv2d_NCHWc_int8(1,  128,  17, 128, 7, 1, 3)
        verify_conv2d_NCHWc_int8(1,  128,  17, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  768,  17, 160, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  160,  17, 160, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  160,  17, 192, 7, 1, 3)
        verify_conv2d_NCHWc_int8(1,  160,  17, 160, 7, 1, 3)
        verify_conv2d_NCHWc_int8(1,  160,  17, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  192,  17, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  192,  17, 192, 7, 1, 3)
        verify_conv2d_NCHWc_int8(1,  192,  17, 320, 3, 2, 0)
        verify_conv2d_NCHWc_int8(1,  192,  17, 192, 3, 2, 0)
        verify_conv2d_NCHWc_int8(1, 1280,   8, 320, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1, 1280,   8, 384, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  384,   8, 384, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  384,   8, 384, 3, 1, 1)
        verify_conv2d_NCHWc_int8(1, 1280,   8, 448, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1,  448,   8, 384, 3, 1, 1)
        verify_conv2d_NCHWc_int8(1, 1280,   8, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1, 2048,   8, 320, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1, 2048,   8, 384, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1, 2048,   8, 448, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1, 2048,   8, 192, 1, 1, 0)
        verify_conv2d_NCHWc_int8(1, 1024,  19,  84, 3, 1, 1)

        # batch > 1
        verify_conv2d_NCHWc_int8(7,   32, 149,  32, 3, 1, 0)
        verify_conv2d_NCHWc_int8(8,   32, 149,  32, 3, 1, 0)
        verify_conv2d_NCHWc_int8(32,  32, 149,  32, 3, 1, 0)

        # Asymmetric padding
        verify_conv2d_NCHWc_int8(1,  32,   35,  64,  7, 2, (0, 0, 1, 1))
        verify_conv2d_NCHWc_int8(1,  64,    8, 128,  3, 1, (3, 3, 2, 2))
        verify_conv2d_NCHWc_int8(1,  64,    8,  64,  1, 1, (1, 2, 2, 1))
        verify_conv2d_NCHWc_int8(1,  64,   17, 192,  1, 1, (1, 2))
        verify_conv2d_NCHWc_int8(1,  64,    8,  64,  3, 1, (3, 1))
        verify_conv2d_NCHWc_int8(1, 128,    8, 384,  3, 1, (0, 2))
        verify_conv2d_NCHWc_int8(1,  64,    8,  64,  1, 1, "VALID")
        verify_conv2d_NCHWc_int8(1, 388,    8,  64,  3, 1, "VALID")
        verify_conv2d_NCHWc_int8(1, 512,   19,  64,  1, 1, "SAME")
        verify_conv2d_NCHWc_int8(1,  64,   16,  32,  2, 1, "SAME")
        verify_conv2d_NCHWc_int8(1,  64,    8,  64,  3, 1, (1, 2, 2, 1), add_relu=True)
        verify_conv2d_NCHWc_int8(1,  64,    8,  64,  5, 2, (1, 3), add_bias=True)
        verify_conv2d_NCHWc_int8(1,  64,   56,  64,  3, 1, "VALID", add_bias=True, add_relu=True)
        verify_conv2d_NCHWc_int8(1,  64,   56,  64, 24, 1, "SAME", add_bias=True, add_relu=True)

        # Conv2d NCHW int8 schedule testing. Internally, it uses NCHWc schedule. So, just
        # performing basic testing - one test for all different scenarios - batch, dilation etc..
        verify_conv2d_nchw_int8(1,  64,  56,  64, 3, 1, 1)
        verify_conv2d_nchw_int8(1, 64, 56, 64, 3, 1, 1, add_relu=True)
        verify_conv2d_nchw_int8(1, 64, 56, 64, 3, 1, 1, dilation=2)
        verify_conv2d_nchw_int8(9, 64, 56, 64, 3, 1, 1)
        verify_conv2d_nchw_int8(4, 4, 4, 4, 4, 4, 4)
        verify_conv2d_nchw_int8(1,   32, 149,  32, 3, 1, 0)
        verify_conv2d_nchw_int8(7,   32, 149,  32, 3, 1, 0)
        verify_conv2d_nchw_int8(1,  32,   35,  64,  7, 2, (0, 0, 1, 1))

def test_conv2d_nhwc():
    with Int8Fallback():
        # Subset of inception v3 expanded (dilation > 1, batch > 1, 'VALID' padding)
        verify_conv2d_NHWC_gemm_int8(1, 3, 299, 32, 3, 2, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 32, 149, 32, 3, 1, 'SAME', dilation=2)
        verify_conv2d_NHWC_gemm_int8(4, 32, 147, 64, 3, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 64, 73, 80, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 80, 73, 192, 3, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 192, 35, 48, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 192, 35, 64, 1, 1, 'VALID')
        verify_conv2d_NHWC_gemm_int8(1, 192, 35, 32, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 48, 35, 64, 5, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 96, 35, 96, 3, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 256, 35, 48, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 256, 35, 64, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 288, 35, 64, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 288, 35, 48, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 96, 35, 96, 3, 2, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 128, 17, 192, 7, 1, 'SAME', dilation=2)
        verify_conv2d_NHWC_gemm_int8(1, 160, 17, 160, 7, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 160, 17, 192, 1, 1, 'VALID')
        verify_conv2d_NHWC_gemm_int8(1, 192, 17, 192, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 768, 5, 128, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 192, 17, 320, 3, 2, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 192, 17, 192, 3, 2, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 1280, 8, 192, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 1280, 8, 384, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 1280, 8, 320, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 1280, 8, 448, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 384, 8, 384, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 384, 8, 384, 3, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 448, 8, 384, 3, 1, 'VALID')
        verify_conv2d_NHWC_gemm_int8(1, 2048, 8, 320, 1, 1, 'SAME')
        verify_conv2d_NHWC_gemm_int8(1, 2048, 8, 448, 1, 1, 'SAME', add_bias=True, add_relu=True)
        verify_conv2d_NHWC_gemm_int8(1, 2048, 8, 192, 1, 1, 'SAME', add_bias=True)

        # Let's also verify that it compiles fine on AArch64 targets
        compile_conv2d_NHWC_gemm_int8_arm(1, 3, 299, 32, 3, 2, 'SAME')


if __name__ == "__main__":
    test_conv2d_nchw()
    test_conv2d_nhwc()

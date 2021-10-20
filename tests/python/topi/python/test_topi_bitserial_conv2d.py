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
from tvm import topi
import tvm.testing
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple
from tvm.contrib.pickle_memoize import memoize


def generate_quantized_np(shape, bits, out_dtype):
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)


def verify_bitserial_conv2d_nchw(
    batch,
    in_size,
    in_channel,
    num_filter,
    kernel,
    stride,
    padding,
    activation_bits,
    weight_bits,
    unipolar,
):
    in_height = in_width = in_size
    input_dtype = "uint32"
    out_dtype = "int32"

    with tvm.target.Target("llvm"):
        A = te.placeholder((batch, in_channel, in_height, in_width), dtype=input_dtype, name="A")
        W = te.placeholder((num_filter, in_channel, kernel, kernel), dtype=input_dtype, name="W")
        B = topi.x86.bitserial_conv2d_nchw(
            A, W, stride, padding, activation_bits, weight_bits, input_dtype, out_dtype, unipolar
        )
        s = topi.x86.schedule_bitserial_conv2d_nchw([B])

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    @memoize("topi.tests.test_topi_bitseral_conv2d_nchw")
    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(a_shape), activation_bits, input_dtype)
        w_np = generate_quantized_np(get_const_tuple(w_shape), weight_bits, input_dtype)
        if unipolar:
            w_ = np.copy(w_np).astype(out_dtype)
            for x in np.nditer(w_, op_flags=["readwrite"]):
                x[...] = 1 if x == 1 else -1
            b_np = tvm.topi.testing.conv2d_nchw_python(a_np.astype(out_dtype), w_, stride, padding)
        else:
            b_np = tvm.topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()

    dev = tvm.cpu(0)
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    func = tvm.build(s, [A, W, B], "llvm")
    func(a, w, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


def verify_bitserial_conv2d_nhwc(
    batch,
    in_size,
    in_channel,
    num_filter,
    kernel,
    stride,
    padding,
    activation_bits,
    weight_bits,
    unipolar,
):
    in_height = in_width = in_size
    input_dtype = "uint32"
    out_dtype = "int32"

    with tvm.target.Target("llvm"):
        A = te.placeholder((batch, in_height, in_width, in_channel), dtype=input_dtype, name="A")
        W = te.placeholder((kernel, kernel, in_channel, num_filter), dtype=input_dtype, name="W")
        B = topi.x86.bitserial_conv2d_nhwc(
            A, W, stride, padding, activation_bits, weight_bits, input_dtype, out_dtype, unipolar
        )
        s = topi.x86.schedule_bitserial_conv2d_nhwc([B])

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    @memoize("topi.tests.test_topi_bitseral_conv2d_nhwc")
    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(a_shape), activation_bits, input_dtype)
        w_np = generate_quantized_np(get_const_tuple(w_shape), weight_bits, input_dtype)
        if unipolar:
            w_ = np.copy(w_np).astype(out_dtype)
            for x in np.nditer(w_, op_flags=["readwrite"]):
                x[...] = 1 if x == 1 else -1
            b_np = tvm.topi.testing.conv2d_nhwc_python(a_np, w_, stride, padding).astype(out_dtype)
        else:
            b_np = tvm.topi.testing.conv2d_nhwc_python(a_np, w_np, stride, padding).astype(
                out_dtype
            )
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()

    dev = tvm.cpu(0)
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    func = tvm.build(s, [A, W, B], "llvm")

    func(a, w, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


def test_bitserial_conv2d():
    in_size = 56
    ic, oc = 64, 64
    k = 3
    stride = 1
    pad = 1
    verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 1, 1, True)
    verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 2, 1, True)
    verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 1, 1, False)
    verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 2, 1, False)
    verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 2, 2, False)

    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 1, 1, True)
    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 1, True)
    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 1, 1, False)
    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 1, False)
    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 2, False)


if __name__ == "__main__":
    test_bitserial_conv2d()

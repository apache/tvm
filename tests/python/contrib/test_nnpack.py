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
import tvm
import tvm.testing
from tvm import te
import numpy as np
import scipy.signal
from tvm.topi.nn.utils import get_pad_tuple
from tvm.contrib import nnpack
import pytest


@tvm.testing.requires_llvm
def test_fully_connected_inference():
    n = 1024
    l = 128
    m = 235
    bias = te.var("bias", dtype="float32")
    A = te.placeholder((l,), name="A")
    B = te.placeholder((m, l), name="B")
    C = nnpack.fully_connected_inference(A, B)
    D = te.compute(C.shape, lambda i: C[i] + bias, name="D")
    s = te.create_schedule(D.op)

    def verify(target="llvm"):
        if not tvm.get_global_func("tvm.contrib.nnpack.fully_connected_inference", True):
            pytest.skip("extern function is not available")
        if not nnpack.is_available():
            pytest.skip("nnpack is not available")

        dev = tvm.cpu(0)
        f = tvm.build(s, [A, B, D, bias], target)
        a = tvm.nd.array(np.random.uniform(size=(l)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(m, l)).astype(B.dtype), dev)
        d = tvm.nd.array(np.zeros((m,), dtype=D.dtype), dev)
        bb = 10.0
        f(a, b, d, bb)
        tvm.testing.assert_allclose(d.numpy(), np.dot(a.numpy(), b.numpy().T) + bb, rtol=1e-5)

    verify()


def np_conv(na, nw, padding, stride=1):
    batch, in_channel, in_height, in_width = na.shape
    _, num_filter, kernel_h, kernel_w = nw.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel_h, kernel_w))
    pad_h = pad_top + pad_bottom
    pad_w = pad_left + pad_right

    out_channel = num_filter
    out_height = (in_height - kernel_h + pad_h) // stride_h + 1
    out_width = (in_width - kernel_w + pad_w) // stride_w + 1
    nb = np.zeros((batch, out_channel, out_height, out_width))
    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad_h > 0 or pad_w > 0:
                    apad = np.zeros((in_height + pad_h, in_width + pad_w))
                    apad[pad_top : pad_top + in_height, pad_left : pad_left + in_width] = na[n, c]
                else:
                    apad = na[n, c]
                out = scipy.signal.convolve2d(apad, np.rot90(np.rot90(nw[f, c])), mode="valid")
                nb[n, f] += out[::stride, ::stride]
    return nb


@tvm.testing.requires_llvm
def test_convolution_inference():
    BATCH = 8
    IH = 48
    IW = 48
    IC = 16
    OC = 16
    K = 3
    PAD = 1
    STRIDE = 1

    OH = (IH + 2 * PAD - K) + 1
    OW = (IW + 2 * PAD - K) + 1
    dshape = (BATCH, IC, IH, IW)
    kshape = (OC, IC, K, K)
    bshape = (OC,)
    oshape = (BATCH, OC, OH, OW)

    data = te.placeholder(dshape, name="data")
    kernel = te.placeholder(kshape, name="kernel")
    bias = te.placeholder(bshape, name="bias")

    def verify(target="llvm", algorithm=nnpack.ConvolutionAlgorithm.AUTO, with_bias=True):
        if not tvm.get_global_func("tvm.contrib.nnpack.fully_connected_inference", True):
            pytest.skip("extern function is not available")
        if not nnpack.is_available():
            pytest.skip("nnpack is not available")

        dev = tvm.cpu(0)
        output = nnpack.convolution_inference(
            data,
            kernel,
            bias if with_bias else None,
            [PAD, PAD, PAD, PAD],
            [STRIDE, STRIDE],
            algorithm=algorithm,
        )
        s = te.create_schedule(output.op)

        f = tvm.build(s, [data, kernel, bias, output], target)

        na = np.random.uniform(size=dshape).astype(data.dtype)
        nb = np.random.uniform(size=kshape).astype(kernel.dtype)
        nc = np.zeros(bshape, dtype=bias.dtype)
        ta = tvm.nd.array(na, dev)
        tb = tvm.nd.array(nb, dev)
        tc = tvm.nd.array(nc, dev)
        td = tvm.nd.array(np.zeros(oshape, dtype=output.dtype), dev)
        f(ta, tb, tc, td)
        nd = np_conv(np.reshape(na, (BATCH, IC, IH, IW)), nb, PAD, STRIDE) + nc.reshape(
            1, bshape[0], 1, 1
        )
        tvm.testing.assert_allclose(td.numpy(), nd.reshape(BATCH, IC, IH, IW), rtol=1e-5)

    for algorithm in [
        nnpack.ConvolutionAlgorithm.AUTO,
        nnpack.ConvolutionAlgorithm.FFT_8x8,
        nnpack.ConvolutionAlgorithm.FFT_16x16,
        nnpack.ConvolutionAlgorithm.WT_8x8,
        nnpack.ConvolutionAlgorithm.IMPLICIT_GEMM,
        nnpack.ConvolutionAlgorithm.WT_8x8_FP16,
    ]:
        for with_bias in [True, False]:
            verify(algorithm=algorithm, with_bias=with_bias)


@tvm.testing.requires_llvm
def test_convolution_inference_without_weight_transform():
    BATCH = 6
    IH = 48
    IW = 48
    IC = 16
    OC = 16
    K = 3
    PAD = 1
    STRIDE = 1

    OH = (IH + 2 * PAD - K) + 1
    OW = (IW + 2 * PAD - K) + 1
    dshape = (BATCH, IC, IH, IW)
    kshape = (OC, IC, K, K)
    bshape = (OC,)
    oshape = (BATCH, OC, OH, OW)

    data = te.placeholder(dshape, name="data")
    kernel = te.placeholder(kshape, name="kernel")
    bias = te.placeholder(bshape, name="bias")

    def verify(target="llvm", algorithm=nnpack.ConvolutionAlgorithm.AUTO, with_bias=True):
        if not tvm.get_global_func("tvm.contrib.nnpack.fully_connected_inference", True):
            pytest.skip("extern function is not available")
        if not nnpack.is_available():
            pytest.skip("nnpack is not available")

        dev = tvm.cpu(0)
        transformed_kernel = nnpack.convolution_inference_weight_transform(
            kernel, algorithm=algorithm
        )
        output = nnpack.convolution_inference_without_weight_transform(
            data,
            transformed_kernel,
            bias if with_bias else None,
            [PAD, PAD, PAD, PAD],
            [STRIDE, STRIDE],
            algorithm=algorithm,
        )

        s = te.create_schedule(output.op)

        f = tvm.build(s, [data, kernel, bias, output], target)

        na = np.random.uniform(size=dshape).astype(data.dtype)
        nb = np.random.uniform(size=kshape).astype(kernel.dtype)
        nc = (
            np.random.uniform(size=bshape).astype(bias.dtype)
            if with_bias
            else np.zeros(bshape, dtype=bias.dtype)
        )
        ta = tvm.nd.array(na, dev)
        tb = tvm.nd.array(nb, dev)
        tc = tvm.nd.array(nc, dev)
        td = tvm.nd.array(np.zeros(oshape, dtype=output.dtype), dev)
        f(ta, tb, tc, td)
        nd = np_conv(np.reshape(na, (BATCH, IC, IH, IW)), nb, PAD, STRIDE) + nc.reshape(
            1, bshape[0], 1, 1
        )
        tvm.testing.assert_allclose(td.numpy(), nd.reshape(BATCH, IC, IH, IW), rtol=1e-5)

    for algorithm in [nnpack.ConvolutionAlgorithm.WT_8x8]:
        for with_bias in [True, False]:
            verify(algorithm=algorithm, with_bias=with_bias)


if __name__ == "__main__":
    tvm.testing.main()

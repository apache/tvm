import tvm
import numpy as np
import scipy.signal
from tvm.contrib import nnpack

def test_fully_connected_output():
    n = 1024
    l = 128
    m = 235
    bias = tvm.var('bias', dtype=tvm.float32)
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((m, l), name='B')
    C = nnpack.fully_connected_output(A, B)
    D = tvm.compute(C.shape, lambda i, j: C[i,j] + bias, name="D")
    s = tvm.create_schedule(D.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.nnpack.fully_connected_output", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A, B, D, bias], target)
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(m, l)).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), ctx)
        bb = 10.0
        f(a, b, d, bb)
        np.testing.assert_allclose(
            d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy().T) + bb, rtol=1e-5)
    verify()


def test_fully_connected_inference():
    n = 1024
    l = 128
    m = 235
    bias = tvm.var('bias', dtype=tvm.float32)
    A = tvm.placeholder((l, ), name='A')
    B = tvm.placeholder((m, l), name='B')
    C = nnpack.fully_connected_inference(A, B)
    D = tvm.compute(C.shape, lambda i: C[i] + bias, name="D")
    s = tvm.create_schedule(D.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.nnpack.fully_connected_inference", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A, B, D, bias], target)
        a = tvm.nd.array(np.random.uniform(size=(l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(m, l)).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros((m, ), dtype=D.dtype), ctx)
        bb = 10.0
        f(a, b, d, bb)
        np.testing.assert_allclose(
            d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy().T) + bb, rtol=1e-5)
    verify()

def np_conv(na, nw, padding, stride=1):
    batch, in_channel, in_height, in_width = na.shape
    _, num_filter, kernel_h, kernel_w = nw.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding * 2
    else:
        pad_h, pad_w = padding
        pad_h *= 2
        pad_w *= 2

    pad_top = int(np.ceil(float(pad_h) / 2))
    pad_bottom = pad_h - pad_top
    pad_left = int(np.ceil(float(pad_w) / 2))
    pad_right = pad_w - pad_left

    out_channel = num_filter
    out_height = (in_height - kernel_h + pad_h) // stride_h + 1
    out_width = (in_width - kernel_w + pad_w) // stride_w + 1
    nb = np.zeros((batch, out_channel, out_height, out_width))
    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad_h > 0:
                    apad = np.zeros((in_height + pad_h, in_width + pad_w))
                    apad[pad_top:-pad_bottom, pad_left:-pad_right] = na[n, c]
                else:
                    apad = na[n, c]
                out = scipy.signal.convolve2d(
                    apad, np.rot90(np.rot90(nw[f, c])), mode='valid')
                nb[n, f] += out[::stride, ::stride]
    return nb

def test_convolution_inference():
    BATCH = 32
    IH = 48
    IW = 48
    IC = 16
    OC = 16
    K = 3
    PAD = 1
    STRIDE = 1

    OH = (IH + 2*PAD - K) + 1
    OW = (IW + 2*PAD - K) + 1
    dshape = (IC, IH, IW)
    kshape = (OC, IC, K, K)
    bshape = (OC, )
    oshape = (OC, OH, OW)

    data = tvm.placeholder(dshape, name='data')
    kernel = tvm.placeholder(kshape, name='kernel')
    bias = tvm.placeholder(bshape, name='bias')
    output = nnpack.convolution_inference(data, kernel, bias,
        [PAD, PAD, PAD, PAD], [STRIDE, STRIDE])
    s = tvm.create_schedule(output.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.nnpack.fully_connected_inference", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [data, kernel, bias, output], target)

        na = np.random.uniform(size=dshape).astype(data.dtype)
        nb = np.random.uniform(size=kshape).astype(kernel.dtype)
        nc = np.zeros(bshape, dtype=bias.dtype)
        ta = tvm.nd.array(na, ctx)
        tb = tvm.nd.array(nb, ctx)
        tc = tvm.nd.array(nc, ctx)
        td = tvm.nd.array(np.zeros(oshape, dtype=output.dtype), ctx)
        f(ta, tb, tc, td)
        nd = np_conv(np.reshape(na, (1, IC, IH, IW)), nb, PAD, STRIDE)
        np.testing.assert_allclose(
            td.asnumpy(), nd.reshape(IC, IH, IW), rtol=1e-5)
    verify()

def test_convolution_output():
    BATCH = 32
    IH = 48
    IW = 48
    IC = 16
    OC = 16
    K = 3
    PAD = 1

    OH = (IH + 2*PAD - K) + 1
    OW = (IW + 2*PAD - K) + 1
    dshape = (BATCH, IC, IH, IW)
    kshape = (OC, IC, K, K)
    bshape = (OC, )
    oshape = (BATCH, OC, OH, OW)

    data = tvm.placeholder(dshape, name='data')
    kernel = tvm.placeholder(kshape, name='kernel')
    bias = tvm.placeholder(bshape, name='bias')
    output = nnpack.convolution_output(data, kernel, bias, [PAD, PAD, PAD, PAD])
    s = tvm.create_schedule(output.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.nnpack.fully_connected_inference", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [data, kernel, bias, output], target)

        na = np.random.uniform(size=dshape).astype(data.dtype)
        nb = np.random.uniform(size=kshape).astype(kernel.dtype)
        nc = np.zeros(bshape, dtype=bias.dtype)
        ta = tvm.nd.array(na, ctx)
        tb = tvm.nd.array(nb, ctx)
        tc = tvm.nd.array(nc, ctx)
        td = tvm.nd.array(np.zeros(oshape, dtype=output.dtype), ctx)
        f(ta, tb, tc, td)
        nd = np_conv(na, nb, PAD)
        np.testing.assert_allclose(
            td.asnumpy(), nd, rtol=1e-5)
    verify()

if __name__ == "__main__":
    import nose
    nose.runmodule()

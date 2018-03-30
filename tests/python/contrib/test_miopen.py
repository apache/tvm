import tvm
from tvm.contrib import miopen
import numpy as np


def test_conv2d():
    in_channel = 3
    out_channel = 64
    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1

    xshape = [1, in_channel, 128, 128]
    if not tvm.module.enabled("rocm"):
        print("skip because rocm is not enabled...")
        return
    if not tvm.get_global_func("tvm.contrib.miopen.conv2d.setup", True):
        print("skip because miopen is not enabled...")
        return
    wshape = (out_channel, in_channel, filter_h, filter_w)

    X = tvm.placeholder(xshape, name='X')
    W = tvm.placeholder(wshape, name='W')
    Y = miopen.conv2d_forward(X,
                              W,
                              stride_h,
                              stride_w,
                              pad_h,
                              pad_w,
                              dilation_h,
                              dilation_w,
                              conv_mode=0)

    yshape = [x.value for x in Y.shape]
    import topi
    with tvm.target.create("rocm -libs=miopen"):
        s = topi.generic.schedule_extern(Y)

    def verify():
        ctx = tvm.rocm(0)
        f = tvm.build(s, [X, W, Y], "rocm", target_host="llvm", name="conv2d")
        x = tvm.nd.array(np.random.uniform(-1, 1, xshape).astype(np.float32), ctx)
        w = tvm.nd.array(np.random.uniform(-1, 1, wshape).astype(np.float32), ctx)
        y = tvm.nd.array(np.random.uniform(-1, 1, yshape).astype(np.float32), ctx)
        f(x, w, y)

        Y_ref = topi.nn.conv2d_nchw(X, W, (stride_h, stride_w), (pad_h, pad_w))
        with tvm.target.rocm():
            s_ref = topi.generic.schedule_conv2d_nchw([Y_ref])
        f_ref = tvm.build(s_ref, [X, W, Y_ref], "rocm")
        y_ref = tvm.nd.array(np.random.uniform(-1, 1, yshape).astype(np.float32), ctx)
        f_ref(x, w, y_ref)
        print("Max abs diff:", np.max(np.abs(y.asnumpy() - y_ref.asnumpy())))
        np.testing.assert_allclose(y.asnumpy(), y_ref.asnumpy(), atol=1e-3)

    verify()


if __name__ == "__main__":
    test_conv2d()

import tvm
from tvm.contrib import cudnn
import numpy as np

def test_conv2d():
    in_channel = 3
    out_channel = 32
    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1

    xshape = [4, 3, 32, 32]
    if not tvm.module.enabled("cuda"):
        print("skip because cuda is not enabled...")
        return
    if not tvm.get_global_func("tvm.contrib.cudnn.conv2d.output_shape", True):
        print("skip because cudnn is not enabled...")
        return
    wshape = cudnn.conv2d_w_shape(in_channel,
                              out_channel,
                              filter_h,
                              filter_w)

    X = tvm.placeholder(xshape, name='X')
    W = tvm.placeholder(wshape, name='W')
    Y = cudnn.conv2d_forward(X,
                             W,
                             stride_h,
                             stride_w,
                             pad_h,
                             pad_w,
                             dilation_h,
                             dilation_w,
                             conv_mode=1,
                             tensor_format=0,
                             algo=1)
    yshape = [x.value for x in Y.shape]
    s =  tvm.create_schedule(Y.op)
    
    def verify():
        ctx = tvm.gpu(0)
        f = tvm.build(s, [X, W, Y], "cuda", target_host="llvm", name="conv2d")
        x = tvm.nd.array(np.random.uniform(-1, 1, xshape).astype(np.float32),
                         ctx)
        w = tvm.nd.array(np.random.uniform(-1, 1, wshape).astype(np.float32),
                         ctx)
        y = tvm.nd.array(np.random.uniform(-1, 1, yshape).astype(np.float32),
                         ctx)
        f(x, w, y)
    
    verify()

def test_softmax():
    mode = "instance"
    dshape = [32, 32]
    
    if not tvm.module.enabled("cuda"):
        print("skip because cuda is not enabled...")
        return
    if not tvm.get_global_func("tvm.contrib.cudnn.conv2d.output_shape", True):
        print("skip because cudnn is not enabled...")
        return

    def verify(alg):
        X = tvm.placeholder(dshape, name='X')
        Y = cudnn.softmax_forward(X,
                                  alg,
                                  mode)
        s =  tvm.create_schedule(Y.op)

        x_np = np.random.uniform(-1, 1, dshape).astype(np.float32)
        
        ctx = tvm.gpu(0)
        f = tvm.build(s, [X, Y], "cuda", target_host="llvm", name="softmax")
        x = tvm.nd.array(x_np, ctx)
        y = tvm.nd.array(np.zeros(dshape, dtype=Y.dtype), ctx)
        f(x, y)

    # testing softmax
    verify("accurate")
    # testing log_softmax
    verify("log")

    
if __name__ == "__main__":
    test_conv2d()
    test_softmax()

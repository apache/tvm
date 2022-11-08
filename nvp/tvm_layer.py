import tvm
from tvm import relay
from tvm.relay.testing import init

def gen_module(layer, layout, input, kernel):
    ## Padding Reference: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/core/kernels/conv_ops.cc#L571
    if layer == 'relu':
        (b, cw, ih, iw) = input
        data = relay.var("data", shape=[b, cw, ih, iw], dtype="float32")
        Trelay = tvm.relay.nn.relu(data)
    elif layer == 'leaky_relu':
        data = relay.var("data", shape=[1, 100], dtype="float32")
        Trelay = tvm.relay.nn.leaky_relu(data)
    elif layer == 'softmax':
        data = relay.var("data", shape=[1, 100], dtype="float32")
        Trelay = tvm.relay.nn.softmax(data)
    elif layer == 'maxpool':
        (b, cw, ih, iw) = input
        (kh, kw) = kernel
        (dh, dw) = (1, 1) # dilation
        stride = 1
        padding = 'same'
        if padding == 'valid':
            pad = (0, 0)
        elif padding == 'same':
            (oh, ow) = (ih, iw)
            pad_row = (oh-1)*stride + (kh-1)*dh + 1 - ih
            pad_col = (ow-1)*stride + (kw-1)*dw + 1 - iw
            pad_l, pad_t = int(pad_row/2), int(pad_col/2)
            pad_r, pad_b = int(pad_row-pad_l), int(pad_col-pad_t)
            pad = (pad_l, pad_t, pad_r, pad_b)
        else:
            raise NotImplementedError("Currently NOT implemented: %s" %(padding))
        data = relay.var("data", shape=[b, cw, oh, ow], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.max_pool2d(data, pool_size=(kh, kw), padding=pad)
    elif layer == 'avgpool':
        (b, cw, ih, iw) = input
        (kh, kw) = kernel
        (dh, dw) = (1, 1) # dilation
        stride = 1
        padding = 'same'
        if padding == 'valid':
            pad = (0, 0)
        elif padding == 'same':
            (oh, ow) = (ih, iw)
            pad_row = (oh-1)*stride + (kh-1)*dh + 1 - ih
            pad_col = (ow-1)*stride + (kw-1)*dw + 1 - iw
            pad_l, pad_t = int(pad_row/2), int(pad_col/2)
            pad_r, pad_b = int(pad_row-pad_l), int(pad_col-pad_t)
            pad = (pad_l, pad_t, pad_r, pad_b)
        else:
            raise NotImplementedError("Currently NOT implemented: %s" %(padding))
        data = relay.var("data", shape=[b, cw, oh, ow], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.avg_pool2d(data, pool_size=(kh, kw), padding=pad)
    elif layer == 'dwconv': # weight(I) == data(C)/groups
        (b, cw, ih, iw) = input
        (kh, kw) = kernel
        (dh, dw) = (1 ,1) # dilation
        stride = 1
        padding = 'same'
        if padding == 'valid':
            pad = (0, 0)
        elif padding == 'same':
            (oh, ow) = (ih, iw)
            pad_row = (oh-1)*stride + (kh-1)*dh + 1 - ih
            pad_col = (ow-1)*stride + (kw-1)*dw + 1 - iw
            pad_l, pad_t = int(pad_row/2), int(pad_col/2)
            pad_r, pad_b = int(pad_row-pad_l), int(pad_col-pad_t)
            pad = (pad_l, pad_t, pad_r, pad_b)
        else:
            raise NotImplementedError("Currently NOT implemented: %s" %(padding))
        # data = relay.var("data", shape=[1, cw, oh+2, ow+2], dtype="float32") #NCHW
        data = relay.var("data", shape=[1, cw, ih, iw], dtype="float32") #NCHW
        weight = relay.var("weight", shape=[cw, 1, kh, kw], dtype="float32") #OIHW
        Trelay = tvm.relay.nn.conv2d(data, weight, strides=stride, padding=pad, groups=cw)
    elif layer == 'dense':
        data = relay.var("data", shape=[5, 10], dtype="float32") #(d_1, d_2, ..., units_in)
        weight = relay.var("weight", shape=[7,10], dtype="float32") #(units, units_in)
        Trelay = tvm.relay.nn.dense(data, weight)
    elif layer == 'corr':
        data1 = relay.var("data", shape=[1, 8, 5, 5], dtype="float32") #NCHW
        data2 = relay.var("data", shape=[1, 8, 5, 5], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.correlation(data1, data2, kernel_size=3, max_displacement=5,
                                    stride1=1, stride2=1, padding=0, is_multiply=True, layout="NCHW")
    else:
        raise NotImplementedError("Currently NOT implemented: %s" %(layer))

    args = relay.analysis.free_vars(Trelay)   # tvm.relay.Var
    # mod = tvm.IRModule.from_expr(Trelay)
    func = relay.Function(args, Trelay)
    # mod = tvm.IRModule.from_expr(func)
    mod, params = init.create_workload(func)
    return mod

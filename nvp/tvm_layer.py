import tvm
from tvm import relay
from tvm.relay.testing import init

def gen_module(model, layer, layout, data, kernel, stride):
    if layout != 'NCHW':
        raise NotImplementedError("Currently NOT implemented layout: %s"%(layout))
    # Pre-defined layers (tvm.nn.**)
    ## Padding Reference: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/core/kernels/conv_ops.cc#L571
    if layer == 'relu':
        (b, cw, h, w) = data
        data = relay.var("data", shape=[b, cw, h, w], dtype="float32")
        Trelay = tvm.relay.nn.relu(data)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'leaky_relu':
        data = relay.var("data", shape=[1, 100], dtype="float32")
        Trelay = tvm.relay.nn.leaky_relu(data)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'softmax':
        data = relay.var("data", shape=[1, 100], dtype="float32")
        Trelay = tvm.relay.nn.softmax(data)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'maxpool':
        (b, cw, h, w) = data
        (kh, kw) = kernel
        (sh, sw) = stride
        pad = get_padding_tuple(data, kernel, padding='same')
        data = relay.var("data", shape=[b, cw, h, w], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.max_pool2d(data, pool_size=(kh, kw), padding=pad)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'maxpool_3x3':
        module = maxpool_3x3(data, (3, 3), stride)
    elif layer == 'avgpool':
        (b, cw, h, w) = data
        (kh, kw) = kernel
        pad = get_padding_tuple(data, kernel, padding='same')
        data = relay.var("data", shape=[b, cw, h, w], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.avg_pool2d(data, pool_size=(kh, kw), padding=pad)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'dwconv': # weight(I) == data(C)/groups
        (b, cw, h, w) = data
        (kh, kw) = kernel
        stride = 1
        pad = get_padding_tuple(data, kernel, padding='same', stride=stride)
        data = relay.var("data", shape=[1, cw, h, w], dtype="float32") #NCHW
        weight = relay.var("weight", shape=[cw, 1, kh, kw], dtype="float32") #OIHW
        Trelay = tvm.relay.nn.conv2d(data, weight, strides=stride, padding=pad, groups=cw)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'dwconv_3x3':
        module = dwconv_3x3(data, kernel, stride)
    elif layer == 'dense':
        data = relay.var("data", shape=[5, 10], dtype="float32") #(d_1, d_2, ..., units_in)
        weight = relay.var("weight", shape=[7,10], dtype="float32") #(units, units_in)
        Trelay = tvm.relay.nn.dense(data, weight)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'corr':
        data1 = relay.var("data", shape=[1, 8, 5, 5], dtype="float32") #NCHW
        data2 = relay.var("data", shape=[1, 8, 5, 5], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.correlation(data1, data2, kernel_size=3, max_displacement=5,
                                    stride1=1, stride2=1, padding=0, is_multiply=True, layout="NCHW")
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    # User-defined layers
    elif layer == 'eadd':
        module = eadd(data)
    else:
        raise NotImplementedError("Currently NOT implemented: %s" %(layer))

    keys = [key for key in module.functions.keys()]
    return module.functions[keys[-1]]


def get_module_from_Trelay(Trelay):
    """Get module from Trelay

    Parameters
    ----------
    Trelay : tvm.relay.expr.Call

    Returns
    -------
    module: tvm.ir.IRModule
    """
    args = relay.analysis.free_vars(Trelay) # tvm.relay.expr.Call --> tvm.relay.Var
    func = relay.Function(args, Trelay)
    module, params = init.create_workload(func)
    return module


def lower_module(model, module):
    """Lower 'RelayOp IRModule' to 'TIR IRModule'

      Parameters
      ----------
      module : IRModule that contains RelayOp only

      Returns
      -------
      module : IRModule that contains TIR only
    """
    LowerTE = tvm._ffi.get_global_func("relay.tec.LowerTE")
    OPT_LEVEL = 2

    if model == "llvm":
      prim_target =tvm.target.Target("llvm")
    elif model == "hexagon":
      prim_target = tvm.target.Target("hexagon")
    elif model == "cuda":
      prim_target = tvm.target.Target("cuda")
    elif model == "x220":
      prim_target = tvm.target.Target("x220")
    else:
      raise NotImplementedError("Currently NOT supported model: %s"%(model))
    ctxt = tvm.transform.PassContext()
    config = tvm.target.make_compilation_config(ctxt, prim_target)
    module = tvm.relay.transform.PlanDevices(config)(module)
    module = tvm.relay.transform.FuseOps(fuse_opt_level=OPT_LEVEL)(module)
    module = tvm.relay.transform.InferType()(module)
    module = LowerTE("default", config)(module)
    return module


def get_padding_tuple(data, kernel, padding, stride=1, dilation=(1, 1)):
    """Get padding tuple

      Returns
      -------
      pad: tuple
    """
    (b, cw, h, w) = data
    (kh, kw) = kernel
    (dh, dw) = dilation
    if padding == 'valid':
        pad = (0, 0)
    elif padding == 'same':
        (oh, ow) = (h, w)
        pad_row = (oh-1)*stride + (kh-1)*dh + 1 - h
        pad_col = (ow-1)*stride + (kw-1)*dw + 1 - w
        pad_l, pad_t = int(pad_row/2), int(pad_col/2)
        pad_r, pad_b = int(pad_row-pad_l), int(pad_col-pad_t)
        pad = (pad_l, pad_t, pad_r, pad_b)
    else:
        raise NotImplementedError("Currently NOT implemented: %s" %(padding))
    return pad

def maxpool_3x3(data, kernel, stride):
    (b, cw, h, w) = data
    (kh, kw) = kernel
    (sh, sw) = stride
    # pad = get_padding_tuple(data, kernel, padding='same')
    # (oh, ow) = ((h - kh + (pad[1]+pad[3])) // sh + 1), ((w - kw + (pad[0]+pad[2])) // sw + 1)

    data = tvm.te.placeholder((b, cw, h, w), name="data")
    data00 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+0], name="data")
    data01 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+0], name="data")
    data02 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+0], name="data")
    data10 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+1], name="data")
    data11 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+1], name="data")
    data12 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+1], name="data")
    data20 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+2], name="data")
    data21 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+2], name="data")
    data22 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+2], name="data")
    output = tvm.te.compute((b, cw, h, w),
        lambda b, c, i, j:
            tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(
                data00[b, c, i, j], data01[b, c, i, j]), data02[b, c, i, j]), 
                data10[b, c, i, j]), data11[b, c, i, j]), data12[b, c, i, j]),
                data20[b, c, i, j]), data21[b, c, i, j]), data22[b, c, i, j])
        , name="output"
    )
    s = tvm.te.create_schedule([output.op])
    s[data00].compute_inline()
    s[data01].compute_inline()
    s[data02].compute_inline()
    s[data10].compute_inline()
    s[data11].compute_inline()
    s[data12].compute_inline()
    s[data20].compute_inline()
    s[data21].compute_inline()
    s[data22].compute_inline()
    module = tvm.lower(s, [data, output])
    return module

def dwconv_3x3(data, kernel, stride):
    if kernel!=(3, 3): raise ValueError("Dwconv_3x3 only supports kernel size with 3x3: %s"%(kernel))
    (b, cw, h, w) = data
    (kh, kw) = kernel
    (sh, sw) = stride
    pad = get_padding_tuple(data, kernel, padding='same')
    (oh, ow) = ((h - kh + (pad[1]+pad[3])) // sh + 1), ((w - kw + (pad[0]+pad[2])) // sw + 1)

    data = tvm.te.placeholder((b, cw, h, w), name="data")
    # data00, data01, data02, data10, data11, data12, data20, data21, data22 = tvm.te.compute((b, cw, oh, ow),
    #     lambda b, c, i, j: (
    #         data[b, c, i*sh+0, j*sw+0], data[b, c, i*sh+1, j*sw+0], data[b, c, i*sh+2, j*sw+0], 
    #         data[b, c, i*sh+0, j*sw+1], data[b, c, i*sh+1, j*sw+1], data[b, c, i*sh+2, j*sw+1], 
    #         data[b, c, i*sh+0, j*sw+2], data[b, c, i*sh+1, j*sw+2], data[b, c, i*sh+2, j*sw+2], 
    #     ), name="data"
    # )
    # data00 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+0], name="data")
    # data01 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+0], name="data")
    # data02 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+0], name="data")
    # data10 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+1], name="data")
    # data11 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+1], name="data")
    # data12 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+1], name="data")
    # data20 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+2], name="data")
    # data21 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+2], name="data")
    # data22 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+2], name="data")
    Reg00 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+0], name="Reg")
    Reg01 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+0], name="Reg")
    Reg02 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+0], name="Reg")
    Reg10 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+1], name="Reg")
    Reg11 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+1], name="Reg")
    Reg12 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+1], name="Reg")
    Reg20 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+2], name="Reg")
    Reg21 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+2], name="Reg")
    Reg22 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+2], name="Reg")
    # Reg00 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data00[b, c, i, j] + Reg__[b, c, i, j], name="Reg")
    # Reg01 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data01[b, c, i, j] + Reg__[b, c, i, j], name="Reg")
    # Reg02 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data02[b, c, i, j] + Reg__[b, c, i, j], name="Reg")
    # Reg10 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data10[b, c, i, j] + Reg__[b, c, i, j], name="Reg")
    # Reg11 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data11[b, c, i, j] + Reg__[b, c, i, j], name="Reg")
    # Reg12 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data12[b, c, i, j] + Reg__[b, c, i, j], name="Reg")
    # Reg20 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data20[b, c, i, j] + Reg__[b, c, i, j], name="Reg")
    # Reg21 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data21[b, c, i, j] + Reg__[b, c, i, j], name="Reg")
    # Reg22 = tvm.te.compute((b, cw, oh, ow), lambda b, c, i, j: data22[b, c, i, j] + Reg__[b, c, i, j], name="Reg")
    # Reg = tvm.te.compute((b, cw, oh, ow),
    #     lambda b, c, i, j:
    #         data00[b, c, i, j] + data01[b, c, i, j] + data02[b, c, i, j] + 
    #         data10[b, c, i, j] + data11[b, c, i, j] + data12[b, c, i, j] + 
    #         data20[b, c, i, j] + data21[b, c, i, j] + data22[b, c, i, j]
    #     , name="Reg"
    # )
    output = tvm.te.compute((b, cw, oh, ow),
        lambda b, c, i, j: (
            Reg00[b, c, i, j] + Reg01[b, c, i, j] + Reg02[b, c, i, j] +
            Reg10[b, c, i, j] + Reg11[b, c, i, j] + Reg12[b, c, i, j] +
            Reg20[b, c, i, j] + Reg21[b, c, i, j] + Reg22[b, c, i, j]
        ), name = "output"
    )
    s = tvm.te.create_schedule([output.op])
    # s[data00].compute_inline()
    # s[data01].compute_inline()
    # s[data02].compute_inline()
    # s[data10].compute_inline()
    # s[data11].compute_inline()
    # s[data12].compute_inline()
    # s[data20].compute_inline()
    # s[data21].compute_inline()
    # s[data22].compute_inline()
    s[Reg00].compute_at(s[output], s[output].op.axis[-1])
    s[Reg01].compute_at(s[output], s[output].op.axis[-1])
    s[Reg02].compute_at(s[output], s[output].op.axis[-1])
    s[Reg10].compute_at(s[output], s[output].op.axis[-1])
    s[Reg11].compute_at(s[output], s[output].op.axis[-1])
    s[Reg12].compute_at(s[output], s[output].op.axis[-1])
    s[Reg20].compute_at(s[output], s[output].op.axis[-1])
    s[Reg21].compute_at(s[output], s[output].op.axis[-1])
    s[Reg22].compute_at(s[output], s[output].op.axis[-1])

    # s[data00].compute_at(s[Reg], s[Reg].op.axis[-1])
    # s[data01].compute_at(s[Reg], s[Reg].op.axis[-1])
    # s[data02].compute_at(s[Reg], s[Reg].op.axis[-1])
    # s[data10].compute_at(s[Reg], s[Reg].op.axis[-1])
    # s[data11].compute_at(s[Reg], s[Reg].op.axis[-1])
    # s[data12].compute_at(s[Reg], s[Reg].op.axis[-1])
    # s[data20].compute_at(s[Reg], s[Reg].op.axis[-1])
    # s[data21].compute_at(s[Reg], s[Reg].op.axis[-1])
    # s[data22].compute_at(s[Reg], s[Reg].op.axis[-1])

    # s[Reg].compute_at(s[output], s[output].op.axis[3])
    module = tvm.lower(s, [data, output])
    return module

def eadd(data):
    if len(data)==4: (vlength, vcount) = (data[0]*data[2]*data[3], data[1])
    elif len(data)==2: (vlength, vcount) = data # vlength: b*h*w, vcount: cw
    else: raise NotImplementedError("Currently NOT supported data format for elemwise-add: %s"%(data))

    data0 = tvm.te.placeholder((vlength, vcount), name="input0")
    data1 = tvm.te.placeholder((vlength, vcount), name="input1")
    weight0 = tvm.te.const(2, "float")
    weight1 = tvm.te.const(2, "float")
    output = tvm.te.compute((vlength, vcount), lambda i, j: data0[i, j]*weight0 + data1[i, j]*weight1, name="output")
    s = tvm.te.create_schedule([output.op])
    module = tvm.lower(s, [data0, data1, output], simple_mode=True)

    """Implementation using tvm.relay functions"""
    # module = tvm.lower(s, [data0, data1, weight0, weight1, output], simple_mode=True)
    # if len(data)==4: (vlength, vcount) = (data[0]*data[2]*data[3], data[1])
    # elif len(data)==2: (vlength, vcount) = data # vlength: b*h*w, vcount: cw
    # else: raise NotImplementedError("Currently NOT supported data format for elemwise-add: %s"%(data))
    # data0 = tvm.relay.var("data", tvm.relay.TensorType([vlength, vcount], dtype="float32"))
    # data1 = tvm.relay.var("data", tvm.relay.TensorType([vlength, vcount], dtype="float32"))
    # import numpy as np
    # weight0 = tvm.relay.Constant(tvm.nd.array(np.array([1], dtype="float32")))
    # weight1 = tvm.relay.Constant(tvm.nd.array(np.array([1], dtype="float32")))
    # A = tvm.relay.multiply(data0, weight0)
    # B = tvm.relay.multiply(data1, weight1)
    # Trelay = tvm.relay.add(A, B)
    # module = get_module_from_Trelay(Trelay)
    # module = lower_module(model, module)
    return module
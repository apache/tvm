import tvm
from tvm import relay
from tvm.relay.testing import init

def gen_module(model, layer, layout, input, kernel):
    if layout != 'NCHW':
        raise NotImplementedError("Currently NOT implemented layout: %s"%(layout))
    # Pre-defined layers (tvm.nn.**)
    ## Padding Reference: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/core/kernels/conv_ops.cc#L571
    if layer == 'relu':
        (b, cw, ih, iw) = input
        data = relay.var("data", shape=[b, cw, ih, iw], dtype="float32")
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
        (b, cw, ih, iw) = input
        (kh, kw) = kernel
        pad = get_padding_tuple(input, kernel, padding='same')
        data = relay.var("data", shape=[b, cw, ih, iw], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.max_pool2d(data, pool_size=(kh, kw), padding=pad)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'avgpool':
        (b, cw, ih, iw) = input
        (kh, kw) = kernel
        pad = get_padding_tuple(input, kernel, padding='same')
        data = relay.var("data", shape=[b, cw, ih, iw], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.avg_pool2d(data, pool_size=(kh, kw), padding=pad)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'dwconv': # weight(I) == data(C)/groups
        (b, cw, ih, iw) = input
        (kh, kw) = kernel
        stride = 1
        pad = get_padding_tuple(input, kernel, padding='same', stride=stride)
        data = relay.var("data", shape=[1, cw, ih, iw], dtype="float32") #NCHW
        weight = relay.var("weight", shape=[cw, 1, kh, kw], dtype="float32") #OIHW
        Trelay = tvm.relay.nn.conv2d(data, weight, strides=stride, padding=pad, groups=cw)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
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
        (b, cw, ih, iw) = input
        data0 = tvm.relay.var("data", tvm.relay.TensorType([b*ih*iw, cw], dtype="float32"))
        data1 = tvm.relay.var("data", tvm.relay.TensorType([b*ih*iw, cw], dtype="float32"))
        weight0 = tvm.relay.var("weight", tvm.relay.TensorType([b*ih*iw, cw], dtype="float32"))
        weight1 = tvm.relay.var("weight", tvm.relay.TensorType([b*ih*iw, cw], dtype="float32"))
        A = tvm.relay.multiply(data0, weight0)
        B = tvm.relay.multiply(data1, weight1)
        Trelay = tvm.relay.add(A, B)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'eadd0':
        if len(input)==4: (vlength, vcount) = (input[0]*input[2]*input[3], input[1])
        elif len(input)==2: (vlength, vcount) = input # vlength: b*ih*iw, vcount: cw
        else: raise NotImplementedError("Currently NOT supported input format for elemwise-add: %s"%(input))
        data0 = tvm.te.placeholder((vlength, vcount), name="input0")
        data1 = tvm.te.placeholder((vlength, vcount), name="input1")
        weight0 = tvm.te.placeholder((vlength, vcount), name="weight0")
        weight1 = tvm.te.placeholder((vlength, vcount), name="weight1")
        output = tvm.te.compute((vlength, vcount), lambda i, j: data0[i, j]*weight0[i, j] + data1[i, j]*weight1[i, j], name="C")
        s = tvm.te.create_schedule([output.op])
        module = tvm.lower(s, [data0, data1, weight0, weight1, output], simple_mode=True)
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


def get_padding_tuple(input, kernel, padding, stride=1, dilation=(1, 1)):
    """Get padding tuple

      Returns
      -------
      pad: tuple
    """
    (b, cw, ih, iw) = input
    (kh, kw) = kernel
    (dh, dw) = dilation
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
    return pad

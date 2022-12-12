import tvm
from tvm import relay
from tvm.relay.testing import init

def gen_module(model, layer, data, layout=None, kernel=None, stride=None, cluster=None):
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
        pad = get_padding_tuple(data, kernel, padding='same', stride=stride)
        data = relay.var("data", shape=[b, cw, h, w], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.max_pool2d(data, pool_size=(kh, kw), padding=pad)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'maxpool_k3':
        module = maxpool_k3(data, (3, 3), stride)
    elif layer == 'maxpool_k3s2':
        module = maxpool_k3s2(data, (3, 3), (2, 2))
    elif layer == 'avgpool':
        (b, cw, h, w) = data
        (kh, kw) = kernel
        pad = get_padding_tuple(data, kernel, padding='same', stride=stride)
        data = relay.var("data", shape=[b, cw, h, w], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.avg_pool2d(data, pool_size=(kh, kw), padding=pad)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'dwconv': # weight(I) == data(C)/groups
        (b, cw, h, w) = data
        (kh, kw) = kernel
        (sh, sw) = stride
        pad = get_padding_tuple(data, kernel, padding='same', stride=stride)
        data = relay.var("data", shape=[1, cw, h, w], dtype="float32") #NCHW
        weight = relay.var("weight", shape=[cw, 1, kh, kw], dtype="float32") #OIHW
        Trelay = tvm.relay.nn.conv2d(data, weight, strides=stride, padding=pad, groups=cw)
        module = get_module_from_Trelay(Trelay)
        module = lower_module(model, module)
    elif layer == 'dwconv_k3':
        module = dwconv_k3(data, kernel, stride)
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
    elif layer == 'upsample':
        module = upsample(data, kernel, layout, method='nearest_neighbor') # TODO: bilinear upsampling
    elif layer == 'upsample_k2':
        module = upsample_k2(data, (2, 2), layout, method='nearest_neighbor') # TODO: bilinear upsampling
    elif layer == 'gap':
        module = global_average_pooling(data, cluster)
    elif layer == 'actv':
        module = actv(data)
    elif layer == 'silu':
        module = silu(data)
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


def get_padding_tuple(data, kernel, padding, stride, dilation=(1, 1)):
    """Get padding tuple

      Returns
      -------
      pad: tuple
    """
    (b, cw, h, w) = data
    (kh, kw) = kernel
    (dh, dw) = dilation
    (sh, sw) = stride
    if padding == 'valid':
        pad = (0, 0)
    elif padding == 'same':
        (oh, ow) = (h, w)
        pad_row = (oh-1)*sh + (kh-1)*dh + 1 - h
        pad_col = (ow-1)*sw + (kw-1)*dw + 1 - w
        pad_l, pad_t = int(pad_row/2), int(pad_col/2)
        pad_r, pad_b = int(pad_row-pad_l), int(pad_col-pad_t)
        pad = (pad_l, pad_t, pad_r, pad_b)
    else:
        raise NotImplementedError("Currently NOT implemented: %s" %(padding))
    return pad

def maxpool_k3(data, kernel, stride):
    (b, cw, h, w) = data
    (kh, kw) = kernel
    (sh, sw) = stride

    data = tvm.te.placeholder((b, cw, h, w), name="data")

    Reg00 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+0], name="Reg00")
    Reg01 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+0], name="Reg01")
    Reg02 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+0], name="Reg02")
    Reg10 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+1], name="Reg10")
    Reg11 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+1], name="Reg11")
    Reg12 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+1], name="Reg12")
    Reg20 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+2], name="Reg20")
    Reg21 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+2], name="Reg21")
    Reg22 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+2], name="Reg22")

    output = tvm.te.compute((b, cw, h, w),
        lambda b, c, i, j:
            tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(
                Reg00[b, c, i, j],  Reg01[b, c, i, j]), Reg02[b, c, i, j]),
                Reg10[b, c, i, j]), Reg11[b, c, i, j]), Reg12[b, c, i, j]),
                Reg20[b, c, i, j]), Reg21[b, c, i, j]), Reg22[b, c, i, j])
        , name="output"
    )
    s = tvm.te.create_schedule([output.op])
    # (N, C, H, W) = s[output].op.axis
    (d0, d1, d2, d3) = sort_axis(s[output].op.axis)

    s[Reg00].compute_at(s[output], d3)
    s[Reg01].compute_at(s[output], d3)
    s[Reg02].compute_at(s[output], d3)
    s[Reg10].compute_at(s[output], d3)
    s[Reg11].compute_at(s[output], d3)
    s[Reg12].compute_at(s[output], d3)
    s[Reg20].compute_at(s[output], d3)
    s[Reg21].compute_at(s[output], d3)
    s[Reg22].compute_at(s[output], d3)
    s[output].reorder(d0, d1, d2, d3)

    module = tvm.lower(s, [data, output])
    return module

def maxpool_k3s2(data, kernel, stride):
    (b, cw, h, w) = data
    (kh, kw) = kernel
    (sh, sw) = stride

    data = tvm.te.placeholder((b, cw, h, w), name="data")

    # Reg00 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+0], name="Reg00")
    # Reg01 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+0], name="Reg01")
    # Reg02 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+0], name="Reg02")
    # max0 = tvm.te.compute((b, cw, h, w), lambda b, c, i ,j: tvm.te.max(tvm.te.max(Reg00, Reg01), Reg02), name="max0")
    max0 = tvm.te.placeholder((b, cw, h, w), name="max0")

    Reg10 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+1], name="Reg10")
    Reg11 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+1], name="Reg11")
    Reg12 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+1], name="Reg12")
    max1 = tvm.te.compute((b, cw, h, w), lambda b, c, i ,j: tvm.te.max(tvm.te.max(Reg10[b, c, i, j], Reg11[b, c, i, j]), Reg12[b, c, i, j]), name="max1")

    Reg20 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+0, j*sw+2], name="Reg20")
    Reg21 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+1, j*sw+2], name="Reg21")
    Reg22 = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i*sh+2, j*sw+2], name="Reg22")
    max2 = tvm.te.compute((b, cw, h, w), lambda b, c, i ,j: tvm.te.max(tvm.te.max(Reg20[b, c, i, j], Reg21[b, c, i, j]), Reg22[b, c, i, j]), name="max2")

    output = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: tvm.te.max(tvm.te.max(max0[b, c, i, j], max1[b, c, i, j]), max2[b, c, i, j]), name="output")

    s = tvm.te.create_schedule([output.op])
    # (N, C, H, W) = s[output].op.axis
    (d0, d1, d2, d3) = sort_axis(s[output].op.axis)

    s[max1].compute_at(s[output], d3)
    s[max2].compute_at(s[output], d3)
    s[Reg10].compute_at(s[output], d3)
    s[Reg11].compute_at(s[output], d3)
    s[Reg12].compute_at(s[output], d3)
    s[Reg20].compute_at(s[output], d3)
    s[Reg21].compute_at(s[output], d3)
    s[Reg22].compute_at(s[output], d3)
    s[output].reorder(d0, d1, d2, d3)

    module = tvm.lower(s, [data, max0, output])
    return module

def dwconv_k3(data, kernel, stride):
    if kernel!=(3, 3): raise ValueError("Dwconv_3x3 only supports kernel size with 3x3: %s"%(kernel))

    (b, cw, h, w) = data
    (kh, kw) = kernel
    (sh, sw) = stride

    data = tvm.te.placeholder((b, cw, h, w), name="data")
    filter = tvm.te.placeholder((b, 1, kh, kw), name='Filter')

    FReg00, FReg01, FReg02, FReg10, FReg11, FReg12, FReg20, FReg21, FReg22 = tvm.te.compute(
        (b, cw), lambda b, c: (
            filter[b, 0, 0, 0], filter[b, 0, 0, 1], filter[b, 0, 0, 2],
            filter[b, 0, 1, 0], filter[b, 0, 1, 1], filter[b, 0, 1, 2],
            filter[b, 0, 2, 0], filter[b, 0, 2, 1], filter[b, 0, 2, 2],
        ), name="FReg"
    )

    Reg00, Reg01, Reg02, Reg10, Reg11, Reg12, Reg20, Reg21, Reg22 = tvm.te.compute(
        (b, cw, h, w), lambda b, c, i, j: (
            data[b, c, i*sh+0, j*sw+0]*FReg00[b, c], data[b, c, i*sh+0, j*sw+1]*FReg01[b, c], data[b, c, i*sh+0, j*sw+2]*FReg02[b, c],
            data[b, c, i*sh+1, j*sw+0]*FReg10[b, c], data[b, c, i*sh+1, j*sw+1]*FReg11[b, c], data[b, c, i*sh+1, j*sw+2]*FReg12[b, c],
            data[b, c, i*sh+2, j*sw+0]*FReg20[b, c], data[b, c, i*sh+2, j*sw+1]*FReg21[b, c], data[b, c, i*sh+2, j*sw+2]*FReg22[b, c],
        ), name="Reg"
    )

    output = tvm.te.compute(
        (b, cw, h, w), lambda b, c, i, j: (
            Reg00[b, c, i, j] + Reg01[b, c, i, j] + Reg02[b, c, i, j]
            + Reg10[b, c, i, j] + Reg11[b, c, i, j] + Reg12[b, c, i, j]
            + Reg20[b, c, i, j] + Reg21[b, c, i, j] + Reg22[b, c, i, j]
        ), name = "output"
    )

    s = tvm.te.create_schedule([output.op])
    (N, C, H, W) = s[output].op.axis
    # (d0, d1, d2, d3) = sort_axis(s[output].op.axis)
    (d0, d1) = (N, C) # channel must be outer-loop because filter is reused within same channel
    (d2, d3) = sort_axis(s[output].op.axis[2:])

    s[FReg00].compute_at(s[output], C)
    s[FReg01].compute_at(s[output], C)
    s[FReg02].compute_at(s[output], C)
    s[FReg10].compute_at(s[output], C)
    s[FReg11].compute_at(s[output], C)
    s[FReg12].compute_at(s[output], C)
    s[FReg20].compute_at(s[output], C)
    s[FReg21].compute_at(s[output], C)
    s[FReg22].compute_at(s[output], C)

    s[Reg00].compute_at(s[output], d3)
    s[Reg01].compute_at(s[output], d3)
    s[Reg02].compute_at(s[output], d3)
    s[Reg10].compute_at(s[output], d3)
    s[Reg11].compute_at(s[output], d3)
    s[Reg12].compute_at(s[output], d3)
    s[Reg20].compute_at(s[output], d3)
    s[Reg21].compute_at(s[output], d3)
    s[Reg22].compute_at(s[output], d3)
    s[output].reorder(d0, d1, d2, d3)

    module = tvm.lower(s, [data, filter, output])
    return module

def eadd(data):
    if len(data) == 4: (vcount, vlength) = (data[0]*data[2]*data[3], data[1])
    elif len(data) == 2: (vcount, vlength) = data # vcount: cw, vlength: b*h*w
    else: raise NotImplementedError("Currently NOT supported data format for elemwise-add: %s"%(data))

    data0 = tvm.te.placeholder((vcount, vlength), name="input0")
    data1 = tvm.te.placeholder((vcount, vlength), name="input1")
    weight0 = tvm.te.placeholder((1, ), name="wReg0")
    weight1 = tvm.te.placeholder((1, ), name="wReg1")
    bias = tvm.te.placeholder((1, ), name="bReg")

    Reg0 = tvm.te.compute((vcount, vlength), lambda i, j: data0[i, j]*weight0[0] + bias[0], name="Reg0")
    Reg1 = tvm.te.compute((vcount, vlength), lambda i, j: data1[i, j]*weight1[0] + Reg0[i, j], name="Reg1")
    output = tvm.te.compute((vcount, vlength), lambda i, j: Reg1[i, j], name="output")
    # output = tvm.te.compute((vcount, vlength), lambda i, j: data0[i, j]*weight0[0] + data1[i, j]*weight1[0], name="output")

    s = tvm.te.create_schedule([output.op])
    (d0, d1) = sort_axis(s[output].op.axis)

    if vlength==8: # fast-path
        s[Reg0].compute_at(s[Reg1], s[Reg1].op.axis[-1])
        s[Reg1].compute_at(s[output], s[output].op.axis[-1])
        s[output].unroll(s[output].op.axis[-1])
    else:
        s[Reg1].compute_at(s[output], s[output].op.axis[0])
        s[Reg0].compute_at(s[output], s[output].op.axis[0])
        (d0, d1) = sort_axis(s[output].op.axis)
        s[output].reorder(d0, d1)

    module = tvm.lower(s, [data0, data1, weight0, weight1, bias, output], simple_mode=True)
    return module

def upsample(data, kernel, layout, method):
    if method == 'nearest_neighbor':
        (b, cw, h, w) = data
        (kh, kw) = kernel

        data = tvm.te.placeholder((b, cw, h, w), name="data")
        Reg = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i, j], name="Reg")
        output = tvm.te.compute((b, cw, h*kh, w*kw), lambda b, c, i, j: Reg[b, c, i//kh, j//kw], name="output")

        s = tvm.te.create_schedule([output.op])
        ho, wo, hi, wi = s[output].tile(x_parent=output.op.axis[-2], y_parent=output.op.axis[-1], x_factor=kh, y_factor=kw)
        s[Reg].compute_at(s[output], wo)
        module = tvm.lower(s, [data, output])
    else:
        raise NotImplementedError("Currently NOT supported upsampling method: %s"%(method))
    return module

def upsample_k2(data, kernel, layout, method):
    if method == 'nearest_neighbor':
        (b, cw, h, w) = data
        (kh, kw) = kernel

        data = tvm.te.placeholder((b, cw, h, w), name="data")
        Reg = tvm.te.compute((b, cw, h, w), lambda b, c, i, j: data[b, c, i, j], name="Reg")
        output = tvm.te.compute((b, cw, h*kh, w*kw), lambda b, c, i, j: Reg[b, c, i//kh, j//kw], name="output")

        s = tvm.te.create_schedule([output.op])
        (N, C, H, W) = s[output].op.axis
        (d0, d1, d2, d3) = sort_axis(s[output].op.axis)

        ho, wo, hi, wi = s[output].tile(x_parent=H, y_parent=W, x_factor=kh, y_factor=kw)
        if d3 == H: # NCWH
            s[output].reorder(N, C, wo, ho, wi, hi)
            s[Reg].compute_at(s[output], ho)
        elif d3 == W: # NCHW
            s[output].reorder(N, C, ho, wo, hi, wi)
            s[Reg].compute_at(s[output], wo)
        else: # NHWC
            s[output].reorder(N, ho, wo, C, hi, wi)
            s[Reg].compute_at(s[output], C)
        s[output].unroll(hi)
        s[output].unroll(wi)
        module = tvm.lower(s, [data, output])
        return module
    else:
        raise NotImplementedError("Currently NOT supported upsampling method: %s"%(method))

def global_average_pooling(data, cluster):
    (b, cw, bh, w) = data

    ## Phase 0) Initialize
    data = tvm.te.placeholder((b, cw, bh, w), name="data")
    multiplier = tvm.te.const(16, "float")
    # multiplier = tvm.te.compute((b, ), lambda b: multiplier, name="multiplier")
    # multiplier = tvm.te.placeholder((1, ), name='multiplier')
    shifter = tvm.te.const(16, "int32")

    ## Phase 1) Intra-lane reduction
    dh = tvm.te.reduce_axis((0, bh), name="dh")
    dw = tvm.te.reduce_axis((0, w), name="dw")
    Reg0 = tvm.te.compute((b, cw), lambda b, c: tvm.te.sum(data[b, c, dh, dw]*multiplier, axis=[dh, dw]), name="Reg0")
    # output0 = tvm.te.compute((b, cw), lambda b, c: Reg0[b, c], name="output0")
    vReg0, vReg1, vReg2 = tvm.te.compute((b, cw),
        lambda b, c: (
            (Reg0[b, c].astype("int")>>16).astype("float"),
            (Reg0[b, c].astype("int")>>8).astype("float"),
            (Reg0[b, c].astype("int")>>1).astype("float"),
        ), name="vReg"
    )
    output0, output1, output2 = tvm.te.compute((b, cw), lambda b, c: (vReg0[b, c], vReg1[b, c], vReg2[b, c]), name="output0")

    ## Phase 2) Inter-lane reduction
    vReg3, vReg4, vReg5 = tvm.te.compute((b, cw), lambda b, c: (output0[b, c], output1[b, c], output2[b, c]), name="vReg")
    output3, output4, output5 = tvm.te.compute((b, cw), lambda b, c: (vReg3[b, c], vReg4[b, c], vReg5[b, c]), name="output1")

    hReg0, hReg1, hReg2 = tvm.te.compute((b, cw, cluster-1), lambda b, c, cs: (output3[b, c], output4[b, c], output5[b, c]), name="hReg")
    Reg1 = tvm.te.compute((b, cw, cluster-1),
        lambda b, c, cs: (
            hReg0[b, c, cs] + hReg1[b, c, cs] + hReg2[b, c, cs]
            # (hReg0[b, c, cs].astype("int")<<16).astype("float") +
            # (hReg1[b, c, cs].astype("int")<<8).astype("float") +
            # (hReg2[b, c, cs].astype("int")<<1).astype("float")
        ), name="Reg1"
    )
    Reg2 = tvm.te.compute((b, cw, cluster-1),
        lambda b, c, cs: (
            output0[b, c] + output1[b, c] + output2[b, c]
            # (output0[b, c].astype("int")<<16).astype("float") +
            # (output1[b, c].astype("int")<<8).astype("float") +
            # (output2[b, c].astype("int")<<1).astype("float")
        ), name="Reg2"
    )
    Reg3 = tvm.te.compute((b, cw, cluster-1), lambda b, c, cs: Reg1[b, c, cs] + Reg2[b, c, cs], name="Reg3")

    vReg6, vReg7, vReg8 = tvm.te.compute((b, cw, cluster-1),
        lambda b, c, cs: (
            (Reg3[b, c, cs].astype("int")>>16).astype("float"),
            (Reg3[b, c, cs].astype("int")>>8).astype("float"),
            (Reg3[b, c, cs].astype("int")>>1).astype("float"),
        ), name="vReg"
    )
    output6, output7, output8 = tvm.te.compute((b, cw, cluster-1), lambda b, c, cs: (vReg6[b, c, cs], vReg7[b, c, cs], vReg8[b, c, cs]), name="output2")

    vReg9, vReg10, vReg11 = tvm.te.compute((b, cw), lambda b, c: (output6[b, c, 0], output7[b, c, 0], output8[b, c, 0]), name="vReg")
    Reg4 = tvm.te.compute((b, cw),
        lambda b, c: (
            vReg9[b, c] + vReg10[b, c] + vReg11[b, c]
            # (vReg9[b, c].astype("int")<<16).astype("float") +
            # (vReg10[b, c].astype("int")<<8).astype("float") +
            # (vReg11[b, c].astype("int")<<1).astype("float")
        ), name="Reg4"
    )
    output = tvm.te.compute((b, cw), lambda b, c: (Reg4[b, c].astype("int")>>shifter).astype("float"), name="output")

    s = tvm.te.create_schedule([output0.op, output1.op, output2.op, output6.op, output7.op, output8.op, output.op])

    s[Reg0].compute_at(s[vReg0], s[vReg0].op.axis[-1])
    s[Reg0].compute_at(s[vReg1], s[vReg1].op.axis[-1])
    s[Reg0].compute_at(s[vReg2], s[vReg2].op.axis[-1])

    s[vReg0].compute_at(s[output0], s[output0].op.axis[-1])
    s[vReg1].compute_at(s[output1], s[output1].op.axis[-1])
    s[vReg2].compute_at(s[output2], s[output2].op.axis[-1])

    s[vReg3].compute_at(s[output3], s[output3].op.axis[-1])
    s[vReg4].compute_at(s[output4], s[output4].op.axis[-1])
    s[vReg5].compute_at(s[output5], s[output5].op.axis[-1])

    # s[output3].compute_at(s[hReg0], s[hReg0].op.axis[1])
    # s[output4].compute_at(s[hReg1], s[hReg1].op.axis[1])
    # s[output5].compute_at(s[hReg2], s[hReg2].op.axis[1])

    s[hReg0].compute_at(s[Reg1], s[Reg1].op.axis[-1])
    s[hReg1].compute_at(s[Reg1], s[Reg1].op.axis[-1])
    s[hReg2].compute_at(s[Reg1], s[Reg1].op.axis[-1])

    s[Reg1].compute_at(s[Reg3], s[Reg3].op.axis[-1])
    s[Reg2].compute_at(s[Reg3], s[Reg3].op.axis[-1])

    s[Reg3].compute_at(s[vReg6], s[vReg6].op.axis[-1])
    s[Reg3].compute_at(s[vReg7], s[vReg7].op.axis[-1])
    s[Reg3].compute_at(s[vReg8], s[vReg8].op.axis[-1])

    s[vReg6].compute_at(s[output6], s[output6].op.axis[-1])
    s[vReg7].compute_at(s[output7], s[output7].op.axis[-1])
    s[vReg8].compute_at(s[output8], s[output8].op.axis[-1])

    s[vReg9].compute_at(s[Reg4], s[Reg4].op.axis[-1])
    s[vReg10].compute_at(s[Reg4], s[Reg4].op.axis[-1])
    s[vReg11].compute_at(s[Reg4], s[Reg4].op.axis[-1])

    s[Reg4].compute_at(s[output], s[output].op.axis[-1])

    module = tvm.lower(s, [data, output0, output1, output2, output])
    return module

def actv(data):
    if len(data) == 4: (vcount, vlength) = (data[0]*data[2]*data[3], data[1])
    elif len(data) == 2: (vcount, vlength) = data # vcount: cw, vlength: b*h*w
    else: raise NotImplementedError("Currently NOT supported data format for elemwise-add: %s"%(data))

    data = tvm.te.placeholder((vcount, vlength), name="data")

    vReg = tvm.te.compute((vcount, vlength), lambda i, j: (tvm.te.sigmoid(data[i, j])), name="vReg")
    output = tvm.te.compute((vcount, vlength), lambda i, j: vReg[i, j], name="output")

    s = tvm.te.create_schedule([output.op])
    if vlength==8: # fast-path
        s[output].unroll(s[output].op.axis[-1])
    s[vReg].compute_at(s[output], s[output].op.axis[-1])
    module = tvm.lower(s, [data, output])
    return module

def silu(data):
    (b, cw, h, w) = data
    data = tvm.te.placeholder((b, cw, h, w), name="data")
    weight = tvm.te.placeholder((b, cw), name="weight")

    wReg = tvm.te.compute(
        (b, cw), lambda b, c: (
            weight(b, c)
        ), name='wReg'
    )
    vReg = tvm.te.compute(
        (b, cw, h, w), lambda b, c, i, j: (
            tvm.te.sigmoid(data[b, c, i, j])
        ), name="vReg"
    )
    output = tvm.te.compute(
        (b, cw, h, w), lambda b, c, i, j: (
            wReg[b, c] * vReg[b, c, i, j]
        ), name="output"
    )

    s = tvm.te.create_schedule([output.op])
    (N, C, H, W) = s[output].op.axis
    s[wReg].compute_at(s[output], C)
    s[vReg].compute_at(s[output], s[output].op.axis[-1])
    module = tvm.lower(s, [data, weight, output])
    return module

def sort_axis(axes):
    iter_ext = [axis.dom.extent for axis in axes]
    index = sorted(range(len(iter_ext)), key=iter_ext.__getitem__)
    max_index = max(index)
    return [axes[i] for i in index] # sorted in ascending order

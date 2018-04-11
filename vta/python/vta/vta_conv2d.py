"""Namespace for supporting packed_conv2d + ewise variant of nnvm."""

from collections import namedtuple

import logging
import tvm
import topi

from nnvm.top import registry as reg, OpPattern
from . import environment as vta


Workload = namedtuple("Conv2DWorkload",
                      ['height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

def packed_conv2d(data,
                  kernel,
                  padding,
                  strides,
                  out_dtype="int32"):
    """ Packed conv2d function.
    """
    if padding[0]:
        pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0], name="pad_data")
    else:
        pad_data = data
    assert len(data.shape) == 5
    assert len(kernel.shape) == 6
    oheight = topi.util.simplify((pad_data.shape[2] - kernel.shape[2]) // strides[0] + 1)
    owidth = topi.util.simplify((pad_data.shape[3] - kernel.shape[3]) // strides[1] + 1)
    oshape = (data.shape[0], kernel.shape[0], oheight, owidth, kernel.shape[4])

    ishape = topi.util.get_const_tuple(data.shape)
    kshape = topi.util.get_const_tuple(kernel.shape)
    assert data.dtype == "int8", data.dtype
    assert kernel.dtype == "int8", kernel.dtype
    d_i = tvm.reduce_axis((0, kshape[2]), name='d_i')
    d_j = tvm.reduce_axis((0, kshape[3]), name='d_j')
    k_o = tvm.reduce_axis((0, ishape[1]), name='k_o')
    k_i = tvm.reduce_axis((0, ishape[-1]), name='k_i')
    hstride, wstride = strides
    res = tvm.compute(
        oshape,
        lambda b, co, i, j, ci: tvm.sum(
            pad_data[b, k_o, i*hstride+d_i, j*wstride+d_j, k_i].astype(out_dtype) *
            kernel[co, k_o, d_i, d_j, ci, k_i].astype(out_dtype),
            axis=[k_o, d_i, d_j, k_i]),
        name="res", tag="packed_conv2d")
    return res


@tvm.register_func("nnvm.compiler.build_target", override=True)
def _build(funcs, target, target_host):
    tvm_t = tvm.target.create(target)
    if tvm_t.device_name == "vta":
        return tvm.build(funcs, target="ext_dev", target_host=target_host)
    elif tvm_t.device_name == "rasp" or tvm_t.device_name == "vta-cpu":
        return tvm.build(funcs, target=target_host)
    return tvm.build(funcs, target=target)


@tvm.register_func("nnvm.compiler.lower", override=True)
def _lower(sch, inputs, func_name, graph):
    import traceback
    # pylint: disable=broad-except
    try:
        f = tvm.lower(sch, inputs, name=func_name)
        if "quantized_conv2d" in func_name:
            logging.info(graph.ir(join_entry_attrs=["shape"]))
    except Exception:
        msg = traceback.format_exc()
        msg += "Error during compile graph\n"
        msg += "--------------------------\n"
        msg += graph.ir(join_entry_attrs=["shape"])
        raise RuntimeError(msg)
    return f if isinstance(
        f, (tvm.container.Array, tuple, list)) else [f]


@reg.register_compute("clip", level=11)
def compute_clip(attrs, inputs, _):
    """ Clip operator.
    """
    x = inputs[0]
    a_min = attrs.get_float("a_min")
    a_max = attrs.get_float("a_max")
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    with tvm.tag_scope(topi.tag.ELEMWISE):
        x = tvm.compute(
            x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
        x = tvm.compute(
            x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x


reg.register_pattern("identity", OpPattern.INJECTIVE, level=11)

@reg.register_compute("quantized_conv2d", level=11)
def compute_quantized_conv2d(attrs, inputs, out):
    """ 2D convolution algorithm.
    """
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    channels = attrs.get_int("channels")
    layout = attrs["layout"]
    out_dtype = attrs['out_type']
    cmp_dtype = 'int32' # compute data type

    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert attrs.get_bool("use_bias") is False
    pack_channel = attrs.get_int("pack_channel")
    if pack_channel != 0:
        assert groups == 1
        return packed_conv2d(inputs[0], inputs[1],
                             padding, strides)
    if groups == 1:
        out = topi.nn.conv2d(inputs[0], inputs[1], strides, padding, out_dtype=cmp_dtype)
    elif groups == get_const_int(inputs[0].shape[1]) and groups == channels:
        out = topi.nn.depthwise_conv2d_nchw(
            inputs[0], inputs[1], strides, padding, out_dtype=cmp_dtype)
    else:
        raise ValueError("not support arbitrary group number for now")

    assert out_dtype == cmp_dtype
    return out


@reg.register_schedule("quantized_conv2d", level=11)
def schedule_quantized_conv2d(attrs, outs, target):
    """ 2D convolution schedule.
    """
    channels = attrs.get_int("channels")
    pack_channel = attrs.get_int("pack_channel")
    if channels != 0 and pack_channel:
        target = tvm.target.create(target)
        if target.device_name == "vta":
            return schedule_packed_conv2d(outs)
        elif target.startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        else:
            raise RuntimeError("not support target %s" % target)
    with tvm.target.create(target):
        return topi.generic.schedule_conv2d_nchw(outs)


def _get_workload(data, pad_data, kernel, output):
    """ Get the workload structure.
    """
    o_shape = topi.util.get_const_tuple(output.shape)
    d_shape = topi.util.get_const_tuple(data.shape)
    k_shape = topi.util.get_const_tuple(kernel.shape)
    o_b, o_c, o_h, o_w, o_blk = o_shape
    i_b, i_c, i_h, i_w, i_blk = d_shape
    k_o, k_i, k_h, k_w, ko_blk, ki_blk = k_shape
    # For now we need to assume that input channel blocking is the same
    # as the output channel blocking
    assert o_blk == i_blk
    # Make sure that dimensions match
    assert o_b == i_b
    assert o_blk == ko_blk
    assert i_blk == ki_blk
    assert k_o == o_c
    assert k_i == i_c
    # Scale the channel size
    i_c *= i_blk
    o_c *= o_blk
    if pad_data is not None:
        p_shape = topi.util.get_const_tuple(pad_data.shape)
        h_pad = (p_shape[2] - d_shape[2]) // 2
        w_pad = (p_shape[3] - d_shape[3]) // 2
    else:
        h_pad, w_pad = 0, 0
    h_str = (i_h + h_pad*2 - k_h) // (o_h - 1)
    w_str = (i_w + w_pad*2 - k_w) // (o_w - 1)
    return Workload(i_h, i_w, i_c, o_c, k_h, k_w, h_pad, w_pad, h_str, w_str)

_WL2PLAN = {}

def schedule_packed_conv2d(outs):
    """ Schedule the packed conv2d.
    """
    assert len(outs) == 1
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
    assert output.dtype == "int8"
    assert output.op.input_tensors[0].dtype == "int32"

    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "packed_conv2d"
            conv2d_res.append(op)

    _traverse(output.op)
    assert len(conv2d_res) == 1
    conv2d_stage = conv2d_res[0].output(0)

    data, kernel = conv2d_stage.op.input_tensors
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None
    wrkld = _get_workload(data, pad_data, kernel, output)

    plan = _WL2PLAN[wrkld]
    env = vta.get_env()

    load_inp = load_wgt = load_out = store_out = env.dma_copy
    alu = env.alu
    gevm = env.gevm

    # schedule1
    oshape = topi.util.get_const_tuple(output.shape)
    s = tvm.create_schedule(output.op)


    # setup pad
    if pad_data is not None:
        cdata = pad_data
        s[pad_data].set_scope(env.inp_scope)
    else:
        cdata = s.cache_read(data, env.inp_scope, [conv2d_stage])
    ckernel = s.cache_read(kernel, env.wgt_scope, [conv2d_stage])
    s[conv2d_stage].set_scope(env.acc_scope)
    # cache read input
    cache_read_ewise = []

    for consumer, tensor in ewise_inputs:
        cache_read_ewise.append(
            s.cache_read(tensor, env.acc_scope, [consumer]))
    # set ewise scope
    for op in ewise_ops:
        s[op].set_scope(env.acc_scope)
        s[op].pragma(s[op].op.axis[0], alu)

    # tile
    oc_factor = (plan.oc_factor if plan.oc_factor
                 else wrkld.out_filter // vta.BLOCK_OUT)
    h_factor = (plan.h_factor if plan.h_factor else oshape[2])
    w_factor = (plan.w_factor if plan.w_factor else oshape[3])

    x_b, x_oc, x_i, x_j, x_ic = s[output].op.axis
    x_oc0, x_oc1 = s[output].split(x_oc, factor=oc_factor)
    x_i0, x_i1 = s[output].split(x_i, factor=h_factor)
    x_j0, x_j1 = s[output].split(x_j, factor=w_factor)
    s[output].reorder(x_b, x_oc0, x_i0, x_j0, x_oc1, x_i1, x_j1, x_ic)
    store_pt = x_j0

    # set all compute scopes
    s[conv2d_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)

    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], load_out)

    # virtual threading along output channel axes
    if plan.oc_nthread:
        _, v_t = s[output].split(x_oc0, factor=plan.oc_nthread)
        s[output].reorder(v_t, x_b)
        s[output].bind(v_t, tvm.thread_axis("cthread"))

    # virtual threading along spatial rows
    if plan.h_nthread:
        _, v_t = s[output].split(x_i0, factor=plan.h_nthread)
        s[output].reorder(v_t, x_b)
        s[output].bind(v_t, tvm.thread_axis("cthread"))

    x_b, x_oc, x_i, x_j, x_ic = s[conv2d_stage].op.axis
    k_o, d_i, d_j, k_i = s[conv2d_stage].op.reduce_axis
    s[conv2d_stage].reorder(k_o, x_j, d_j, d_i, x_oc, x_i, x_ic, k_i)

    if plan.ko_factor:
        k_o, _ = s[conv2d_stage].split(k_o, factor=plan.ko_factor)
        s[cdata].compute_at(s[conv2d_stage], k_o)
        s[ckernel].compute_at(s[conv2d_stage], k_o)

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[0], load_inp)
    s[ckernel].pragma(s[ckernel].op.axis[0], load_wgt)
    s[conv2d_stage].tensorize(x_ic, gevm)
    s[output].pragma(x_oc1, store_out)
    return s


class Conv2DSchedule(object):
    """ 2D convolution schedule object.
    """
    def __init__(self,
                 oc_factor,
                 ko_factor=1,
                 h_factor=1,
                 w_factor=0,
                 oc_nthread=0,
                 h_nthread=0):
        self.oc_factor = oc_factor
        self.ko_factor = ko_factor
        self.h_factor = h_factor
        self.w_factor = w_factor
        self.oc_nthread = oc_nthread
        self.h_nthread = h_nthread

Schedule = Conv2DSchedule

# ResNet18 workloads
RESNET = {
    # Workloads of resnet18 on imagenet
    0: Workload(224, 224, 16, 64, 7, 7, 3, 3, 2, 2),
    1: Workload(56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    2: Workload(56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    3: Workload(56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    4: Workload(56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    5: Workload(28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    6: Workload(28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    7: Workload(28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    8: Workload(14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    9: Workload(14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    10: Workload(14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    11: Workload(7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
}

# Serial schedule
RESNET_SERIAL = {
    RESNET[0]: Schedule(oc_factor=1, ko_factor=1, h_factor=4, w_factor=56),
    RESNET[1]: Schedule(oc_factor=2, ko_factor=1, h_factor=14, w_factor=0),
    RESNET[2]: Schedule(oc_factor=4, ko_factor=4, h_factor=8, w_factor=0),
    RESNET[3]: Schedule(oc_factor=4, ko_factor=1, h_factor=14, w_factor=0),
    RESNET[4]: Schedule(oc_factor=8, ko_factor=1, h_factor=4, w_factor=0),
    RESNET[5]: Schedule(oc_factor=8, ko_factor=1, h_factor=7, w_factor=0),
    RESNET[6]: Schedule(oc_factor=8, ko_factor=1, h_factor=14, w_factor=0),
    RESNET[7]: Schedule(oc_factor=16, ko_factor=1, h_factor=7, w_factor=0),
    RESNET[8]: Schedule(oc_factor=8, ko_factor=1, h_factor=7, w_factor=0),
    RESNET[9]: Schedule(oc_factor=8, ko_factor=1, h_factor=7, w_factor=0),
    RESNET[10]: Schedule(oc_factor=16, ko_factor=1, h_factor=7, w_factor=0),
    RESNET[11]: Schedule(oc_factor=8, ko_factor=1, h_factor=7, w_factor=0),
}

# Latency hiding schedule
RESNET_OPT = {
    RESNET[0]: Schedule(oc_factor=1, ko_factor=1, h_factor=4, w_factor=56),
    RESNET[1]: Schedule(oc_factor=2, ko_factor=1, h_factor=7, h_nthread=2),
    RESNET[2]: Schedule(oc_factor=4, ko_factor=2, h_factor=4, w_factor=0, h_nthread=2),
    RESNET[3]: Schedule(oc_factor=4, ko_factor=1, h_factor=7, w_factor=0, h_nthread=2),
    RESNET[4]: Schedule(oc_factor=4, ko_factor=1, h_factor=7, h_nthread=2),
    RESNET[5]: Schedule(oc_factor=4, ko_factor=1, h_factor=7, w_factor=0, h_nthread=2),
    RESNET[6]: Schedule(oc_factor=4, ko_factor=1, h_factor=7, w_factor=0, oc_nthread=2),
    RESNET[7]: Schedule(oc_factor=8, ko_factor=1, h_factor=7, w_factor=0, oc_nthread=2),
    RESNET[8]: Schedule(oc_factor=4, ko_factor=1, h_factor=7, w_factor=0, oc_nthread=2),
    RESNET[9]: Schedule(oc_factor=4, ko_factor=1, h_factor=7, w_factor=0, oc_nthread=2),
    RESNET[10]: Schedule(oc_factor=8, ko_factor=1, h_factor=7, w_factor=0, oc_nthread=2),
    RESNET[11]: Schedule(oc_factor=4, ko_factor=1, h_factor=7, w_factor=0, oc_nthread=2),
}

_WL2PLAN = RESNET_OPT

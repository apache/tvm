"""Namespace for supporting packed_conv2d + ewise variant of nnvm."""
from __future__ import absolute_import as _abs

from collections import namedtuple

import logging
import tvm
import topi

from nnvm.top import registry as reg, OpPattern
from nnvm.top import nn as _nn
from ..environment import get_env


Workload = namedtuple("Conv2DWorkload",
                      ['batch', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

def find_schedules(layer, vt_only=False, best_only=False):
    """ Returns a schedule for a given a layer.

    Parameters
    ----------
    layer : Workload
        Convolutional layer description.
    vt_only : Boolean
        Produce a schedule plan with virtual threading.
    best_only : Boolean
        Return the "best" schedule plan.

    Returns
    -------
    fil_sched : list
        List of valid schedules.

    """
    # pylint: disable=too-many-nested-blocks
    env = get_env()

    # Helper function to get factors
    def _find_factors(n):
        factors = []
        for f in range(1, n + 1):
            if n % f == 0:
                factors.append(f)
        return factors

    def _get_data_movement_byte(schedule, layer):
        """ Estimate data movement in bytes for the schedule plan
        """
        env = get_env()
        b_f = schedule.b_factor
        h_f = schedule.h_factor
        w_f = schedule.w_factor
        ci_f = schedule.ic_factor
        co_f = schedule.oc_factor
        # Derive data movement
        inp_elem_sizeb = env.BATCH * env.BLOCK_IN * env.INP_WIDTH
        wgt_elem_sizeb = env.BLOCK_IN * env.BLOCK_OUT * env.WGT_WIDTH
        out_elem_sizeb = env.BATCH * env.BLOCK_OUT * env.OUT_WIDTH
        input_tile_elems = b_f * \
                ((h_f - 1) * layer.hstride + layer.hkernel) * \
                ((w_f - 1) * layer.wstride + layer.wkernel) * ci_f
        weight_tile_elems = layer.hkernel * layer.wkernel * ci_f
        output_tile_elems = b_f * h_f * w_f * co_f
        # Derive tiling factors
        b_factor = layer.batch // (b_f * env.BATCH)
        h_factor = (layer.height // layer.hstride) // h_f
        w_factor = (layer.width // layer.wstride) // w_f
        ci_factor = layer.in_filter // (ci_f * env.BLOCK_IN)
        co_factor = layer.out_filter // (co_f * env.BLOCK_OUT)
        # Compute input transaction count
        input_xfers = b_factor * h_factor * w_factor * co_factor * ci_factor
        weight_xfers = b_factor * h_factor * w_factor * co_factor * ci_factor
        output_xfers = b_factor * h_factor * w_factor * co_factor
        # Compute total transfer sizes
        input_xfer_byte = input_tile_elems * input_xfers * inp_elem_sizeb // 8
        weight_xfer_byte = weight_tile_elems * weight_xfers * wgt_elem_sizeb // 8
        output_xfer_byte = output_tile_elems * output_xfers * out_elem_sizeb // 8
        total_xfer_byte = input_xfer_byte + weight_xfer_byte + output_xfer_byte
        return total_xfer_byte

    # Scheduling exploration
    batch_factors = _find_factors(layer.batch // env.BATCH)
    height_factors = _find_factors(layer.height // layer.hstride)
    width_factors = _find_factors(layer.width // layer.wstride)
    cin_factors = _find_factors(layer.in_filter // env.BLOCK_IN)
    cout_factors = _find_factors(layer.out_filter // env.BLOCK_OUT)
    ht_factors = [1, 2]
    cot_factors = [1, 2]

    # Explore schedules
    schedules = []
    for b_f in batch_factors:
        for h_f in height_factors:
            for w_f in width_factors:
                for ci_f in cin_factors:
                    for co_f in cout_factors:
                        # FIXME: 2D load pattern matching imposes restrictions on schedule
                        valid = (w_f == layer.width // layer.wstride) or \
                                (w_f != layer.width // layer.wstride and co_f == 1) and \
                                ci_f == 1
                        if valid:
                            schedules.append([b_f, h_f, w_f, ci_f, co_f])

    # Filter the schedules that wouldn't work in the available BRAM sizes
    inp_elem_sizeb = env.BATCH * env.BLOCK_IN * env.INP_WIDTH
    wgt_elem_sizeb = env.BLOCK_IN * env.BLOCK_OUT * env.WGT_WIDTH
    out_elem_sizeb = env.BATCH * env.BLOCK_OUT * env.OUT_WIDTH
    inp_brams_sizeb = env.INP_BUFF_SIZE * 8
    wgt_brams_sizeb = env.WGT_BUFF_SIZE * 8
    out_brams_sizeb = env.OUT_BUFF_SIZE * 8
    fil_sched = []
    xfer_size = []
    for sched in schedules:
        b_f, h_f, w_f, ci_f, co_f = sched
        for h_t in ht_factors:
            for co_t in cot_factors:
                # Make sure to filter cases where we apply threading on two axes
                # or cases where the threading factors for h and co are not
                # factors of h and co
                if (h_t == 2 and co_t == 2) or (h_f % h_t != 0) or (co_f % co_t != 0):
                    continue
                # Adjust tile sizes if threading is applied
                h_f //= h_t
                co_f //= co_t
                # Derive tile sizes
                input_tile_elems = b_f * \
                        ((h_f - 1) * layer.hstride + layer.hkernel) * \
                        ((w_f - 1) * layer.wstride + layer.wkernel) * ci_f
                weight_tile_elems = layer.hkernel * layer.wkernel * ci_f * co_f
                output_tile_elems = b_f * h_f * w_f * co_f

                # Derive valid schedule filter
                valid = True
                # If in vitrual-threaded mode, only allow for threaded plans
                valid &= (vt_only and (h_t == 2 or co_t == 2)) or not vt_only
                # Check that we don't exceed input/weight/output capacity
                valid &= input_tile_elems * inp_elem_sizeb <= inp_brams_sizeb // (co_t * h_t)
                valid &= weight_tile_elems * wgt_elem_sizeb <= wgt_brams_sizeb
                valid &= output_tile_elems * out_elem_sizeb <= out_brams_sizeb // (co_t * h_t)
                # Make sure that we don't write to the same acc location within 2 consecutive cycles
                valid &= h_f > 2 and w_f > 2
                # TODO: check that we don't exceed instruction or micro-op count

                if valid:
                    schedule = Schedule(b_factor=b_f, oc_factor=co_f, ic_factor=ci_f, h_factor=h_f,
                                        w_factor=w_f, oc_nthread=co_t, h_nthread=h_t)
                    fil_sched.append(schedule)
                    xfer_size.append(_get_data_movement_byte(schedule, layer))

    if best_only:
        return [fil_sched[xfer_size.index(min(xfer_size))]]
    return fil_sched

def packed_conv2d(data,
                  kernel,
                  padding,
                  strides,
                  out_dtype="int32"):
    """ Packed conv2d function.
    """
    if padding[0]:
        pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0], name="pad_data")
    else:
        pad_data = data
    assert len(data.shape) == 6
    assert len(kernel.shape) == 6
    oheight = topi.util.simplify((pad_data.shape[2] - kernel.shape[2]) // strides[0] + 1)
    owidth = topi.util.simplify((pad_data.shape[3] - kernel.shape[3]) // strides[1] + 1)
    oshape = (data.shape[0], kernel.shape[0], oheight, owidth, data.shape[4], kernel.shape[4])

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
        lambda b_o, c_o, i, j, b_i, c_i: tvm.sum(
            pad_data[b_o, k_o, i*hstride+d_i, j*wstride+d_j, b_i, k_i].astype(out_dtype) *
            kernel[c_o, k_o, d_i, d_j, c_i, k_i].astype(out_dtype),
            axis=[k_o, d_i, d_j, k_i]),
        name="res", tag="packed_conv2d")
    return res

@tvm.register_func("nnvm.compiler.build_target", override=True)
def _build(funcs, target, target_host):
    tvm_t = tvm.target.create(target)
    if tvm_t.device_name == "vta":
        return tvm.build(funcs, target="ext_dev", target_host=target_host)
    elif tvm_t.device_name == "rasp" or tvm_t.device_name == "vtacpu":
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


@reg.register_compute("clip", level=15)
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

# override to force partition at copy
reg.register_pattern("copy", OpPattern.INJECTIVE, level=15)

def is_packed_layout(layout):
    """Check if layout is packed layout"""
    if layout == "NCHW":
        return False
    if "n" in layout and "c" in layout:
        return True
    return False

@reg.register_alter_op_layout("conv2d", level=15)
def alter_conv2d_layout(attrs, inputs, out):
    layout = attrs['layout']
    if is_packed_layout(layout):
        return None
    return _nn.alter_conv2d_layout(attrs, inputs, out)


@reg.register_compute("conv2d", level=15)
def compute_conv2d(attrs, inputs, out):
    """ 2D convolution algorithm.
    """
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs["layout"]
    out_dtype = attrs['out_dtype']
    assert dilation == (1, 1), "not support dilate now"
    if is_packed_layout(layout):
        assert groups == 1
        return packed_conv2d(inputs[0], inputs[1],
                             padding, strides, out_dtype=out_dtype)
    return _nn.compute_conv2d(attrs, inputs, out)


@reg.register_schedule("conv2d", level=15)
def schedule_conv2d(attrs, outs, target):
    """ 2D convolution schedule.
    """
    layout = attrs["layout"]

    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "vta":
            return schedule_packed_conv2d(outs)
        elif str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        else:
            raise RuntimeError("not support target %s" % target)
    return _nn.schedule_conv2d(attrs, outs, target)


def _get_workload(data, pad_data, kernel, output):
    """ Get the workload structure.
    """
    o_shape = topi.util.get_const_tuple(output.shape)
    d_shape = topi.util.get_const_tuple(data.shape)
    k_shape = topi.util.get_const_tuple(kernel.shape)
    o_b, o_c, o_h, o_w, ob_blk, o_blk = o_shape
    i_b, i_c, i_h, i_w, ib_blk, i_blk = d_shape
    k_o, k_i, k_h, k_w, ko_blk, ki_blk = k_shape
    # For now we need to assume that input channel blocking is the same
    # as the output channel blocking
    assert o_blk == i_blk
    assert ob_blk == ib_blk
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
    return Workload(i_b, i_h, i_w, i_c, o_c, k_h, k_w, h_pad, w_pad, h_str, w_str)

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
    if wrkld in _WL2PLAN:
        plan = _WL2PLAN[wrkld]
    else:
        plan = find_schedules(wrkld, vt_only=True, best_only=True)[0]
        logging.info("Trying to find plan for %s", wrkld)
    env = get_env()

    load_inp = load_wgt = load_out = store_out = env.dma_copy
    alu = env.alu
    gemm = env.gemm

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
                 else plan.out_filter // env.BLOCK_OUT)
    h_factor = (plan.h_factor if plan.h_factor else oshape[2])
    w_factor = (plan.w_factor if plan.w_factor else oshape[3])

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    x_co0, x_co1 = s[output].split(x_co, factor=oc_factor)
    x_i0, x_i1 = s[output].split(x_i, factor=h_factor)
    x_j0, x_j1 = s[output].split(x_j, factor=w_factor)
    s[output].reorder(x_bo, x_i0, x_co0, x_j0, x_co1, x_i1, x_j1, x_bi, x_ci)
    store_pt = x_j0

    # set all compute scopes
    s[conv2d_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)

    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], load_out)

    # virtual threading along output channel axes
    if plan.oc_nthread > 1:
        _, v_t = s[output].split(x_co0, factor=plan.oc_nthread)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, tvm.thread_axis("cthread"))

    # virtual threading along spatial rows
    if plan.h_nthread > 1:
        _, v_t = s[output].split(x_i0, factor=plan.h_nthread)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, tvm.thread_axis("cthread"))

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[conv2d_stage].op.axis
    k_o, d_i, d_j, k_i = s[conv2d_stage].op.reduce_axis
    s[conv2d_stage].reorder(x_bo, k_o, x_j, d_j, d_i, x_co, x_i, x_bi, x_ci, k_i)

    if plan.ic_factor:
        k_o, _ = s[conv2d_stage].split(k_o, factor=plan.ic_factor)
        s[cdata].compute_at(s[conv2d_stage], k_o)
        s[ckernel].compute_at(s[conv2d_stage], k_o)

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[0], load_inp)
    s[ckernel].pragma(s[ckernel].op.axis[0], load_wgt)
    s[conv2d_stage].tensorize(x_bi, gemm)
    s[output].pragma(x_co1, store_out)
    return s

class Conv2DSchedule(object):
    """ 2D convolution schedule object.
    """
    def __init__(self,
                 b_factor=1,
                 oc_factor=1,
                 ic_factor=1,
                 h_factor=1,
                 w_factor=0,
                 oc_nthread=0,
                 h_nthread=0,
                 debug_sync=False):
        self.b_factor = b_factor
        self.oc_factor = oc_factor
        self.ic_factor = ic_factor
        self.h_factor = h_factor
        self.w_factor = w_factor
        self.oc_nthread = oc_nthread
        self.h_nthread = h_nthread
        self.debug_sync = debug_sync

    def __str__(self):
        return "{}.{}.{}.{}.{}.{}.{}".format(
            self.b_factor, self.oc_factor, self.ic_factor,
            self.h_factor, self.w_factor,
            self.oc_nthread, self.h_nthread)

Schedule = Conv2DSchedule

# Layer description of the ResNet18
RESNET = {
    0: Workload(1, 224, 224, 16, 64, 7, 7, 3, 3, 2, 2),
    1: Workload(1, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    2: Workload(1, 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    3: Workload(1, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    4: Workload(1, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    5: Workload(1, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    6: Workload(1, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    7: Workload(1, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    8: Workload(1, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    9: Workload(1, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    10: Workload(1, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    11: Workload(1, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
}

for idx in RESNET:
    scheds = find_schedules(RESNET[idx], vt_only=True, best_only=True)[0]
    _WL2PLAN[RESNET[idx]] = scheds

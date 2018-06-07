# pylint: disable=invalid-name,unused-variable,invalid-name
"""QConv2D schedule on x86"""
import tvm
from .. import generic, tag
from .. import nn
from ..nn.util import infer_pad, infer_stride
from topi.util import simplify, get_const_int
from ..nn.qconv2d import qconv2d as _qconv2d, _get_schedule
from ..nn.qconv2d import QuantizedSpatialPackNCHW, QuantizedSpatialPackNHWC
from ..nn.qconv2d import _WORKLOADS, _SCH_TO_DECL_FUNC_QUANT
from ..nn.qconv2d import _get_workload


# TODO grab the number from autotuner
_QUANTIZED_SCHEDULES_NCHW = [
    # resnet
    QuantizedSpatialPackNCHW(2, 2, 8, 1, 1),
    QuantizedSpatialPackNCHW(1, 4, 8, 4, 1),
    QuantizedSpatialPackNCHW(1, 4, 8, 1, 16),
    QuantizedSpatialPackNCHW(1, 4, 8, 4, 8),
    QuantizedSpatialPackNCHW(1, 7, 8, 3, 8),
    QuantizedSpatialPackNCHW(1, 2, 8, 1, 8),
    QuantizedSpatialPackNCHW(2, 1, 8, 1, 4),
    QuantizedSpatialPackNCHW(1, 7, 8, 1, 1),
    QuantizedSpatialPackNCHW(1, 1, 8, 1, 16),
    QuantizedSpatialPackNCHW(1, 1, 8, 1, 8),
    QuantizedSpatialPackNCHW(1, 1, 8, 1, 16),
]

_QUANTIZED_SCHEDULES_NHWC = [
    # resnet
    QuantizedSpatialPackNHWC(2, 2, 8, 1, 1),
    QuantizedSpatialPackNHWC(1, 4, 8, 4, 1),
    QuantizedSpatialPackNHWC(1, 4, 8, 1, 16),
    QuantizedSpatialPackNHWC(1, 4, 8, 4, 8),
    QuantizedSpatialPackNHWC(1, 7, 8, 3, 8),
    QuantizedSpatialPackNHWC(1, 2, 8, 1, 8),
    QuantizedSpatialPackNHWC(2, 1, 8, 1, 4),
    QuantizedSpatialPackNHWC(1, 7, 8, 1, 1),
    QuantizedSpatialPackNHWC(1, 1, 8, 1, 16),
    QuantizedSpatialPackNHWC(1, 1, 8, 1, 8),
    QuantizedSpatialPackNHWC(1, 1, 8, 1, 16),
]

@_get_schedule.register("cpu")
def _get_schedule_qconv2d(wkl, layout):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = _WORKLOADS.index(wkl)
    if layout == "NCHW":
        sch = _QUANTIZED_SCHEDULES_NCHW[idx]
    elif layout == "NHWC":
        sch = _QUANTIZED_SCHEDULES_NHWC[idx]
    return sch


@_qconv2d.register("cpu")
def _declaration_qconv2d(data, kernel, stride, padding,  activation_bits, weight_bits, layout='NCHW', 
           pack_dtype=None, out_dtype=None, dorefa=False):
    if out_dtype is None:
        out_dtype = data.dtype
    assert data.shape[0].value == 1, "only support batch size=1 convolution on rasp"
    assert layout == "NCHW" or layout == "NHWC", "only support layouts NCHW and NHWC"

    wkl = _get_workload(data, kernel, stride, padding, out_dtype, layout)
    sch = _get_schedule(wkl, layout)
    return _SCH_TO_DECL_FUNC_QUANT[type(sch)](data, kernel, stride, padding, activation_bits, weight_bits, 
                                              pack_dtype, out_dtype, dorefa)

@generic.schedule_qconv2d_nchw.register(["cpu"])
@generic.schedule_qconv2d_nhwc.register(["cpu"])
def schedule_qconv2d(outs):
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        output = op.output(0)
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag) or 'elemwise' in op.tag or 'uquantize' in op.tag:
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        elif 'spatial_qconv_nchw' in op.tag or 'spatial_qconv_nhwc' in op.tag :
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel_q = kernel_vec.op.input_tensors[0]
            kernel = kernel_q.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[0]
            data_q = data_vec.op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data_q.op.input_tensors[0]
            if "QuantizeInput" in kernel.op.name:
                # Need to go up 1 further, from the combine in bitpack
                kernel = kernel.op.input_tensors[0]
            if "QuantizeInput" in data.op.name:
                # Need to go up 1 further, from the combine in bitpack
                data = data.op.input_tensors[0]

            if 'spatial_qconv_nchw' in op.tag:
                _schedule_spatial_conv2d_nchw(s, data, data_q, data_pad, data_vec,
                                        kernel, kernel_q, kernel_vec,
                                        conv_out, output, outs[0])
            elif 'spatial_qconv_nhwc' in op.tag:
                _schedule_spatial_conv2d_nhwc(s, data, data_q, data_pad, data_vec,
                                        kernel, kernel_q, kernel_vec,
                                        conv_out, output, outs[0])
        else:
            kernel = op.input_tensors[1]
            data_q = op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data_q.op.input_tensors[0]
            if 'conv2d_nchw_q' in op.tag:
                _schedule_conv2d_nchw_q(s, data, data_q, data_pad, kernel, output)
            elif 'conv2d_nhwc_q' in op.tag:
                _schedule_conv2d_nhwc_q(s, data, data_q, data_pad, kernel, output)


    traverse(outs[0].op)
    return s


def _schedule_spatial_conv2d_nchw(s, data, data_q, data_pad, data_vec, kernel, kernel_q, kernel_vec, conv_out, output, last):
    IB, _, CI, IH, IW = data_q.shape
    KB, CO, _, KH, KW = kernel_q.shape
    _, _, OH, OW = output.shape

    # Infer padding and stride
    if data_pad is None:
        padding = (0, 0)
        TH, TW = IH, IW
    else:
        _, _, _, TH, TW = data_pad.shape
        hpad = get_const_int((TH - IH) // 2)
        wpad = get_const_int((TW - IW) // 2)
        padding = (hpad, wpad)

    hstride = get_const_int((TH - KH) // (OH - 1))
    wstride = get_const_int((TW - KW) // (OW - 1))
    stride = (hstride, wstride)
    
    wkl = _get_workload(data, kernel, stride, padding, output.dtype, "NCHW")
    sch = _get_schedule(wkl, "NCHW")
    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    ba = sch.ba
    bc = sch.bc
    
    CC = s.cache_write(conv_out, "global")

    n, co, oh, ow, vh, vw, vc = s[conv_out].op.axis
    s[conv_out].vectorize(vc)

    s[CC].compute_at(s[conv_out], ow)
    n, co, oh, ow, vh, vw, vc = s[CC].op.axis
    ci, dh, dw, b1, b2 = s[CC].op.reduce_axis
    s[CC].reorder(ci, dh, vh, dw, vw, b1, b2, vc)
    s[CC].unroll(b1)
    s[CC].unroll(b2)
    s[CC].vectorize(vc)

    ##### Schedule A
    if data_pad is not None:
        s[data_pad].compute_inline()

    _, h, _, _, _, _ , vw = s[data_vec].op.axis
    s[data_vec].vectorize(vw)
    if ba == 1:
        oaxis = h
        paxis = h
    else:
        oh, ih = s[data_vec].split(h, ba)
        oaxis = oh
        paxis = ih

    s[data_vec].parallel(paxis)
    s[data_vec].pragma(oaxis, "parallel_launch_point")
    s[data_vec].pragma(paxis, "parallel_stride_pattern")
    s[data_vec].pragma(oaxis, "parallel_barrier_when_finish")


    ##### Schedule B
    co, _, _, _, _, vc = s[kernel_vec].op.axis
    s[kernel_vec].vectorize(vc)
    if bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[kernel_vec].split(co, bc)
        oaxis = oco
        paxis = ico

    s[kernel_vec].parallel(paxis)
    s[kernel_vec].pragma(oaxis, "parallel_launch_point")
    s[kernel_vec].pragma(paxis, "parallel_stride_pattern")
    s[kernel_vec].pragma(oaxis, "parallel_barrier_when_finish")


    ##### Schedule C
    n, co, h, w = s[last].op.axis
    co, vc = s[last].split(co, VC)
    oh, ow, vh, vw = s[last].tile(h, w, VH, VW)
    s[last].reorder(n, co, oh, ow, vh, vw, vc)
    if last != output:
        s[output].compute_inline()
    s[conv_out].compute_at(s[last], ow)

    if bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[last].split(co, bc)
        oaxis = oco
        paxis = ico

    s[last].parallel(paxis)
    s[last].pragma(oaxis, "parallel_launch_point")
    s[last].pragma(paxis, "parallel_stride_pattern")
    s[last].pragma(oaxis, "parallel_barrier_when_finish")

    return s

def _schedule_spatial_conv2d_nhwc(s, data, data_q, data_pad, data_vec,
                            kernel, kernel_q, kernel_vec,
                            conv_out, output, last):
    # no stride and padding info here
    _, IH, IW, CI, IB = data_q.shape
    KH, KW, _, CO, KB = kernel_q.shape
    _, OH, OW, _ = output.shape
    # Infer padding and stride
    if data_pad is None:
        padding = (0, 0)
        TH, TW = IH, IW
    else:
        _, TH, TW, _, _ = data_pad.shape
        hpad = get_const_int((TH - IH) // 2)
        wpad = get_const_int((TW - IW) // 2)
        padding = (hpad, wpad)

    hstride = get_const_int((TH - KH) // (OH - 1))
    wstride = get_const_int((TW - KW) // (OW - 1))
    stride = (hstride, wstride)

    wkl = _get_workload(data, kernel, stride, padding, last.dtype, "NHWC")
    sch = _get_schedule(wkl, "NHWC")
    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    ba = sch.ba
    bc = sch.bc

    ##### Schedule data packing
    if data_pad is not None:
        s[data_pad].compute_inline()

    _, h, _, _, _, _ , _ = s[data_vec].op.axis
    if ba == 1:
        oaxis = h
        paxis = h
    else:
        oh, ih = s[data_vec].split(h, ba)
        oaxis = oh
        paxis = ih
    s[data_vec].parallel(paxis)
    s[data_vec].pragma(oaxis, "parallel_launch_point")
    s[data_vec].pragma(paxis, "parallel_stride_pattern")
    s[data_vec].pragma(oaxis, "parallel_barrier_when_finish")


    ##### Schedule kernel packing
    co, _, _, _, _, _ = s[kernel_vec].op.axis
    if bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[kernel_vec].split(co, bc)
        oaxis = oco
        paxis = ico

    s[kernel_vec].parallel(paxis)
    s[kernel_vec].pragma(oaxis, "parallel_launch_point")
    s[kernel_vec].pragma(paxis, "parallel_stride_pattern")
    s[kernel_vec].pragma(oaxis, "parallel_barrier_when_finish")


    ##### Schedule Convolution
    n, oh, ow, co, vh, vw, vc = s[conv_out].op.axis
    dh, dw, ci, b1, b2 = s[conv_out].op.reduce_axis

    s[conv_out].reorder(n, oh, ow, co, vh, vw, dh, dw, ci, vc, b1, b2)

    s[conv_out].unroll(b1)
    s[conv_out].unroll(b2)
    s[conv_out].vectorize(vc)

    # # Schedule output
    n, h, w, co = s[last].op.axis
    co, vc = s[last].split(co, VC)
    oh, ow, vh, vw = s[last].tile(h, w, VH, VW)
    s[last].reorder(n, oh, ow, co, vh, vw, vc)
    s[last].vectorize(vc)
    if last != output:
        s[output].compute_inline()
    s[conv_out].compute_at(s[last], ow)

    if bc == 1:
        oaxis = oh
        paxis = oh
    else:
        oho, iho = s[last].split(oh, bc)
        oaxis = oho
        paxis = iho

    s[last].parallel(paxis)
    s[last].pragma(oaxis, "parallel_launch_point")
    s[last].pragma(paxis, "parallel_stride_pattern")
    s[last].pragma(oaxis, "parallel_barrier_when_finish")

    return s

# Very simple schedules
def schedule_qconv2d_nchw(outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        if 'qconv2d_nchw' in op.tag:
            output = op.output(0)
            kernel = op.input_tensors[1]
            data_q = op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data_q.op.input_tensors[0]

            # Schedule for padding
            n_pad, c_pad, b_pad, h_pad, w_pad = data_pad.op.axis
            pad_fused = s[data_pad].fuse(n_pad, c_pad)
            s[data_pad].parallel(pad_fused)

            # Schedule for convolution
            nn, ff, yy, xx = s[output].op.axis
            rc, ry, rx, b2, b1 = s[output].op.reduce_axis

            # Tiling
            yo, xo, yi, xi = s[output].tile(yy, xx, 4, 4)
            fused = s[output].fuse(nn, ff)
            s[output].reorder(fused,  rc, yo, xo, ry, rx, yi, b1, b2, xi)
            # Vectorize, unroll, parallel
            s[output].vectorize(xi)
            s[output].unroll(b1)
            s[output].unroll(b2)
            s[output].parallel(fused)
    
    traverse(outs[0].op)
    return s

def schedule_qconv2d_nhwc(outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        if 'qconv2d_nhwc' in op.tag:
            output = op.output(0)
            kernel = op.input_tensors[1]
            data_q = op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data_q.op.input_tensors[0]

            # Schedule for padding
            n_pad, h_pad, w_pad, c_pad, b_pad = data_pad.op.axis
            pad_fused = s[data_pad].fuse(n_pad, h_pad)
            s[data_pad].parallel(pad_fused)

            # Schedule for convolution
            nn, yy, xx, ff = s[output].op.axis
            ry, rx, rc, b1, b2 = s[output].op.reduce_axis

            # Tiling
            xo, fo, xi, fi = s[output].tile(xx, ff, 4, 4)
            fused = s[output].fuse(nn, yy)
            s[output].reorder(fused, xo, fo, ry, rx, xi, rc, b1, b2, fi)
            # Vectorize, unroll, parallel
            s[output].vectorize(fi)
            s[output].unroll(b1)
            s[output].unroll(b2)
            s[output].parallel(fused)
    traverse(outs[0].op)
    return s

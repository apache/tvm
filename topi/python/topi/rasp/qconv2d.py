# pylint: disable=invalid-name,unused-variable,invalid-name
"""QConv2D schedule on raspberry pi"""
from __future__ import absolute_import as _abs
import tvm
from tvm import target as _target
from .. import tag
from ..nn.qconv2d import qconv2d as _qconv2d, _get_schedule
from ..nn.qconv2d import RaspQuantizedSpatialPack, QuantizedSpatialPackNCHW, QuantizedSpatialPackNHWC
from ..nn.qconv2d import _WORKLOADS, _SCH_TO_DECL_FUNC_QUANT
from ..nn.qconv2d import _get_workload
from ..nn.util import infer_pad, infer_stride
from ..util import simplify, get_const_int

from .. import generic

# TODO grab the number from autotuner
_QUANTIZED_SCHEDULES = [
    RaspQuantizedSpatialPack(2, 2, 8, 1, 1, False, 8),
    RaspQuantizedSpatialPack(1, 4, 8, 4, 1, False, 8),
    RaspQuantizedSpatialPack(1, 4, 8, 1, 16, False, 8),
    RaspQuantizedSpatialPack(1, 4, 8, 4, 8, False, 8),
    RaspQuantizedSpatialPack(1, 7, 8, 3, 8, False, 16),
    RaspQuantizedSpatialPack(1, 2, 8, 1, 8, False, 16),
    RaspQuantizedSpatialPack(2, 1, 8, 1, 4, False, 16),
    RaspQuantizedSpatialPack(1, 7, 8, 1, 1, True, 16),
    RaspQuantizedSpatialPack(1, 1, 8, 1, 16, True, 16),
    RaspQuantizedSpatialPack(1, 1, 8, 1, 8, True, 16),
    RaspQuantizedSpatialPack(1, 1, 8, 1, 16, True, 16),
]

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


@_get_schedule.register("rasp")
def _get_schedule_qconv2d(wkl, layout):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = _WORKLOADS.index(wkl)
    if layout == "NCHW":
        sch = _QUANTIZED_SCHEDULES_NCHW[idx]
    elif layout == "NHWC":
        sch = _QUANTIZED_SCHEDULES_NHWC[idx]
    return sch


@_qconv2d.register("rasp")
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

# TODO: is there a better way to share these with x86?

@generic.schedule_qconv2d_nchw.register(["rasp"])
@generic.schedule_qconv2d_nhwc.register(["rasp"])
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

            # Need to go up 1 further, from the combine in bitpack
            if "QuantizeInput" in kernel.op.name:
                kernel = kernel.op.input_tensors[0]
            if "QuantizeInput" in data.op.name:
                data = data.op.input_tensors[0]

            if 'spatial_qconv_nchw' in op.tag:
                _schedule_spatial_conv2d_nchw(s, data, data_q, data_pad, data_vec,
                                        kernel, kernel_q, kernel_vec,
                                        conv_out, output, outs[0])
            elif 'spatial_qconv_nhwc' in op.tag:
                _schedule_spatial_conv2d_nhwc(s, data, data_q, data_pad, data_vec,
                                        kernel, kernel_q, kernel_vec,
                                        conv_out, output, outs[0])
        
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

    wkl = _get_workload(data, kernel, stride, padding, last.dtype, "NCHW")
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
    return s
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

    wkl = _get_workload(data, kernel, stride, padding, output.dtype, "NHWC")
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

####### ARM SPECIFIC #######
def _spatial_pack_nhwc(data, kernel, stride, padding, activation_bits, weight_bits, out_dtype):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    print (out_dtype)
    wkl = _get_workload(data, kernel, stride, padding, out_dtype, "NHWC")
    sch = _get_schedule(wkl)
    VH = sch.vh
    VW = sch.vw
    VC = sch.vc

    data_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=3, pack_type='uint8')
    kernel_vec = kernel_vec_spatial_pack_nhwc(kernel, weight_bits, VC)
    N, H, W, IB, CI = data_q.shape
    OCO, KH, KW, KB, VC, _ = kernel_vec.shape

    CO = OCO * VC
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    HCAT, WCAT = KH-1, KW-1


    PAD_H = H + 2*HPAD
    PAD_W = W + 2*WPAD
    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1
    dvshape = (N, PAD_H//(VH*HSTR), PAD_W//(VW*WSTR), VH*HSTR+HCAT, VW*WSTR+WCAT, IB, CI)
    ovshape = (1, OH // VH, OW // VW, CO // VC, VH, VW, VC)
    oshape = (1, OH, OW, CO)

    if (HPAD != 0 and WPAD != 0):
        data_pad = pad(data_q, (0, HPAD, WPAD, 0, 0), name="data_pad")
    else:
        data_pad = data_q

    data_vec = tvm.compute(dvshape, lambda n, h, w, vh, vw, b, ci: \
        data_pad[n][h*VH*HSTR+vh][w*VW*WSTR+vw][b][ci], name='data_vec')
    
    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')
    ib = tvm.reduce_axis((0, IB), name='ib')
    kb = tvm.reduce_axis((0, KB), name='kb')

    def _conv(n, h, w, co, vh, vw, vc):
        return tvm.sum( 
            (tvm.popcount(kernel_vec[co, dh, dw, kb, vc, ci] & 
                data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ib, ci]).astype('int16') 
            << (kb + ib).astype('int16')), axis=[dh, dw, kb, ib, ci])

    conv = tvm.compute(ovshape, _conv, name='conv')

    return tvm.compute(oshape, lambda n, h, w, co:
        conv[n][h//VH][w//VW][co//VC][h%VH][w%VW][co%VC].astype(out_dtype),
        name='output_vec', tag='spatial_qconv_nhwc')

def intrin_popcount(m, k_i, w_b, x_b):
    type = 'uint8'
    w = tvm.placeholder((w_b, m, k_i), dtype=type, name='w')
    x = tvm.placeholder((x_b, k_i,), dtype=type, name='x')
    k = tvm.reduce_axis((0, k_i), name='k')
    bw = tvm.reduce_axis((0, w_b), name='bw')
    bx = tvm.reduce_axis((0, x_b), name='bx')
    z = tvm.compute((m,), lambda i:
                    tvm.sum(tvm.popcount(w[bw, i, k].astype('uint16') & x[bx, k].astype('uint16')) << (bw+bx).astype('uint16'),
                     axis=[bw, bx, k]), name='z')

    Wb = tvm.decl_buffer(w.shape, w.dtype,
                        name="W",
                        offset_factor=k_i,
                        strides=[tvm.var('ldw'), tvm.var('ldw'), 1]) 
    Xb = tvm.decl_buffer(x.shape, x.dtype,
                        name="X",
                        offset_factor=k_i,
                        strides=[tvm.var('ldw'), 1])

            
    def intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]
        vpadd_id = tvm.const(647, 'uint32')
        vpadalu_id = tvm.const(646, 'uint32')
        args_1 = tvm.const(1, 'uint32')
        args_2 = tvm.const(2, 'uint32')
    
        def instr(index):
            irb = tvm.ir_builder.create()
            if index == 1:
                irb.emit(zz.vstore(0, tvm.const(0, 'uint16x8')))
            else:
                cnts8 = [None] * 8
                cnts4 = [None] * 4
                cnts2 = [None] * 2
                for bw in range(w_b):
                    for bx in range(x_b):
                        if k_i == 16:
                            for i in range(m):
                                ands = ww.vload([bw, i, 0], 'uint8x16') & xx.vload([bx, 0], 'uint8x16')
                                cnts = tvm.popcount(ands)
                                upper_half = tvm.call_pure_intrin('uint8x8', 'vectorhigh', cnts)
                                lower_half = tvm.call_pure_intrin('uint8x8', 'vectorlow', cnts)
                                cnts8[i] = upper_half + lower_half
                            for i in range(m/2):
                                cnts4[i] = tvm.call_pure_intrin('uint8x8', 'llvm_intrin', vpadd_id, args_1, cnts8[i*2], cnts8[i*2+1])
                            for i in range(m/4):
                                cnts2[i] = tvm.call_pure_intrin('uint8x8', 'llvm_intrin', vpadd_id, args_1, cnts4[i*2], cnts4[i*2+1])
                            cnts = tvm.call_pure_intrin('uint8x16', 'vectorcombine', cnts2[0], cnts2[1])
                            shifted_cnts = cnts << (bw+bx)
                            out = tvm.call_pure_intrin('uint16x8', 'llvm_intrin', vpadalu_id, args_2, zz.vload(0, 'uint16x8'), shifted_cnts)
                        else: # ki ==8
                            for i in range(m):
                                ands = ww.vload([bw, i, 0], 'uint8x8') & xx.vload([bx, 0], 'uint8x8')
                                cnts8[i] = tvm.popcount(ands)
                            for i in range(m/2):
                                cnts4[i] = tvm.call_pure_intrin('uint8x8', 'llvm_intrin', vpadd_id, args_1, cnts8[i*2], cnts8[i*2+1])
                            for i in range(m/4):
                                cnts2[i] = tvm.call_pure_intrin('uint8x8', 'llvm_intrin', vpadd_id, args_1, cnts4[i*2], cnts4[i*2+1])
                            cnts = tvm.call_pure_intrin('uint8x16', 'vectorcombine', cnts2[0], cnts2[1])
                            shifted_cnts = cnts << (bw+bx)
                            out = tvm.call_pure_intrin('uint16x8', 'llvm_intrin', vpadalu_id, args_2, zz.vload(0, 'uint16x8'), shifted_cnts)
                        irb.emit(zz.vstore(0, out))
            return irb.get()
        # body, reset, update
        return instr(0), instr(1), instr(2)
    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(z.op, intrin_func, binds={w: Wb, x:Xb})


# ARM specific schedule that using custom microkernel
def arm_schedule_spatial_conv2d_nhwc(s, data, data_q, data_pad, data_vec,
                kernel, kernel_q, kernel_vec, conv_out, output, last):
    # no stride and padding info here
    _, H, W, IB, CI = data_q.shape
    KH, KW, KB, _, CO = kernel_q.shape
    KB = get_const_int(KB)
    IB = get_const_int(IB)

    if data_pad is None:
        padding = (0,0)
        _, in_h, in_w, _ , _ = data_q.shape
        kern_h, kern_w, _, _ = kernel.shape
        _, out_h, out_w, _ = output.shape
        hstride = (in_h - kern_h) // (out_h - 1)
        wstride = (in_w - kern_w) // (out_w - 1)
        stride = get_const_int(hstride), get_const_int(wstride)
    else:
        _, in_h, in_w, _, _ = data_q.shape
        _, pad_h, pad_w, _, _ = data_pad.shape
        hpad = (pad_h - in_h) // 2
        wpad = (pad_w - in_w) // 2
        padding = get_const_int(hpad), get_const_int(wpad)

        _, in_h, in_w, _, _ = data_pad.shape
        kern_h, kern_w, _, _ = kernel.shape
        _, out_h, out_w, _ = output.shape
        hstride = (in_h - kern_h) // (out_h - 1)
        wstride = (in_w - kern_w) // (out_w - 1)
        stride = get_const_int(hstride), get_const_int(wstride)

    wkl = _get_workload(data, kernel, stride, padding, output.dtype, "NHWC")
    sch = _get_schedule(wkl, "NHWC")

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    ba = sch.ba
    bc = sch.bc

    ##### Schedule data packing
    if data_pad is not None:
        s[data_pad].compute_inline()

    _, h, _, _, _, _, _ = s[data_vec].op.axis
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
    dh, dw, kb, ib, ci = s[conv_out].op.reduce_axis
    
    kfactor = sch.kfactor
    if sch.split_ci:
        oci, ici = s[conv_out].split(ci, kfactor)
        s[conv_out].reorder(n, oh, ow, co, vh, vw, dh, dw, oci, kb, ib, vc, ici)
    else:
        s[conv_out].reorder(n, oh, ow, co, vh, vw, dh, dw, kb, ib, vc, ci)
   
    pc = intrin_popcount(8, kfactor, KB, IB)
    s[conv_out].tensorize(kb, pc)

    n, h, w, co = s[last].op.axis
    co, vc = s[last].split(co, VC)
    oh, ow, vh, vw = s[last].tile(h, w, VH, VW)
    s[last].reorder(n, oh, ow, co, vc, vh, vw)
    s[last].vectorize(vw)
    if last != output:
        s[last].compute_inline()
    
    s[conv_out].compute_at(s[last], ow)
    if co == 1:
        oaxis = oh
        paxis = oh
    else:
        oho, iho = s[last].split(oh, bc)
        oaxis = oho
        paxis = iho

    s[last].parallel(paxis)
    s = s.normalize()
    return s


# @generic.schedule_qconv2d_nhwc.register(["rasp"])
def schedule_qconv2d_nhwc(outs):
    s = tvm.create_schedule([x.op for x in outs])
    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'spatial_qconv_nhwc' in op.tag:
            # print "spatial"
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[0]
            kernel_q = kernel_vec.op.input_tensors[0]
            kernel = kernel_q.op.input_tensors[0]
            if "QuantizeInput" in kernel.op.name:
                # Need to go up 1 further, from the combine in bitpack
                kernel = kernel.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[1]
            data_q = data_vec.op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data_q.op.input_tensors[0]
            if "QuantizeInput" in data.op.name:
                # Need to go up 1 further, from the combine in bitpack
                data = data.op.input_tensors[0]

            _schedule_spatial_conv2d_nhwc(s, data, data_q, data_pad, data_vec,
                                            kernel, kernel_q, kernel_vec, conv_out, output, outs[0])

    traverse(outs[0].op)
    return s

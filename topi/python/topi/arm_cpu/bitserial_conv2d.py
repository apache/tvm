# pylint: disable=invalid-name,unused-variable,invalid-name
"""Bitserial conv2d schedule on raspberry pi"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm
from .. import tag
from ..nn.pad import pad
from ..nn.bitserial_conv2d import bitserial_conv2d, _get_schedule, _get_workload, bitpack
from ..nn.bitserial_conv2d import SpatialPackNCHW, _WORKLOADS, spatial_pack_nchw
from ..nn.util import get_pad_tuple
from ..util import get_const_int
from .. import generic

RaspSpatialPack = namedtuple('SpatialPack',
                             ['vh', 'vw', 'vc', 'ba', 'bc', 'split_ci', 'kfactor'])

_QUANTIZED_SCHEDULES_NHWC = [
    RaspSpatialPack(2, 2, 8, 1, 1, False, 8),
    RaspSpatialPack(1, 4, 8, 4, 1, False, 8),
    RaspSpatialPack(1, 4, 8, 1, 16, False, 8),
    RaspSpatialPack(1, 4, 8, 4, 8, False, 8),
    RaspSpatialPack(1, 7, 8, 3, 8, False, 16),
    RaspSpatialPack(1, 2, 8, 1, 8, False, 16),
    RaspSpatialPack(2, 1, 8, 1, 4, False, 16),
    RaspSpatialPack(1, 7, 8, 1, 1, True, 16),
    RaspSpatialPack(1, 1, 8, 1, 16, True, 16),
    RaspSpatialPack(1, 1, 8, 1, 8, True, 16),
    RaspSpatialPack(1, 1, 8, 1, 16, True, 16),
]

_QUANTIZED_SCHEDULES_NCHW = [
    # resnet
    SpatialPackNCHW(2, 2, 8, 1, 1),
    SpatialPackNCHW(1, 4, 8, 4, 1),
    SpatialPackNCHW(1, 4, 8, 1, 16),
    SpatialPackNCHW(1, 4, 8, 4, 8),
    SpatialPackNCHW(1, 7, 8, 3, 8),
    SpatialPackNCHW(1, 2, 8, 1, 8),
    SpatialPackNCHW(2, 1, 8, 1, 4),
    SpatialPackNCHW(1, 7, 8, 1, 1),
    SpatialPackNCHW(1, 1, 8, 1, 16),
    SpatialPackNCHW(1, 1, 8, 1, 8),
    SpatialPackNCHW(1, 1, 8, 1, 16),
]

@_get_schedule.register("arm_cpu")
def _get_schedule_bitserial_conv2d(wkl, layout):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = _WORKLOADS.index(wkl)
    if layout == "NCHW":
        sch = _QUANTIZED_SCHEDULES_NCHW[idx]
    elif layout == "NHWC":
        sch = _QUANTIZED_SCHEDULES_NHWC[idx]
    return sch


@bitserial_conv2d.register("arm_cpu")
def _declaration_bitserial_conv2d(data, kernel, stride, padding, activation_bits, weight_bits,
                                  layout='NCHW', pack_dtype=None, out_dtype=None, dorefa=False):
    if out_dtype is None:
        out_dtype = data.dtype
    assert data.shape[0].value == 1, "only support batch size=1 convolution on rasp"
    assert layout == "NCHW" or layout == "NHWC", "only support layouts NCHW and NHWC"
    if dorefa:
        assert layout == "NCHW", "Cannot support dorea with NHWC layout yet"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype, layout)
    sch = _get_schedule(wkl, layout)
    if layout == "NCHW":
        return spatial_pack_nchw(data, kernel, stride, padding, activation_bits, weight_bits,
                                 pack_dtype=pack_dtype, out_dtype=out_dtype, dorefa=dorefa)
    return _spatial_pack_nhwc(data, kernel, stride, padding, activation_bits,
                              weight_bits, out_dtype)

def _kernel_vec_spatial_pack_nhwc(kernel, kernel_bits, VC):
    kernel_q = bitpack(kernel, kernel_bits, pack_axis=2, bit_axis=2, pack_type='uint8')
    KH, KW, KB, CI, CO = kernel_q.shape
    kvshape = (CO//VC, KH, KW, KB, VC, CI)
    return tvm.compute(kvshape, lambda co, dh, dw, b, vc, ci: \
        kernel_q[dh][dw][b][ci][co*VC+vc], name='kernel_vec')

def _spatial_pack_nhwc(data, kernel, stride, padding, activation_bits, weight_bits, out_dtype):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype, "NHWC")
    sch = _get_schedule(wkl, "NHWC")
    VH = sch.vh
    VW = sch.vw
    VC = sch.vc

    data_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=3, pack_type='uint8')
    kernel_vec = _kernel_vec_spatial_pack_nhwc(kernel, weight_bits, VC)
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
        return tvm.sum((tvm.popcount(
            kernel_vec[co, dh, dw, kb, vc, ci].astype('uint16') &
            data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ib, ci].astype('uint16'))
                        << (kb + ib).astype('uint16')), axis=[dh, dw, kb, ib, ci])

    conv = tvm.compute(ovshape, _conv, name='conv')

    return tvm.compute(oshape, lambda n, h, w, co:
                       conv[n][h//VH][w//VW][co//VC][h%VH][w%VW][co%VC].astype(out_dtype),
                       name='output_vec', tag='spatial_bitserial_conv_nhwc')

def _intrin_popcount(m, k_i, w_b, x_b):
    dtype = 'uint8'
    w = tvm.placeholder((w_b, m, k_i), dtype=dtype, name='w')
    x = tvm.placeholder((x_b, k_i,), dtype=dtype, name='x')
    k = tvm.reduce_axis((0, k_i), name='k')
    bw = tvm.reduce_axis((0, w_b), name='bw')
    bx = tvm.reduce_axis((0, x_b), name='bx')
    z = tvm.compute((m,), lambda i:
                    tvm.sum(tvm.popcount(w[bw, i, k].astype('uint16') &
                                         x[bx, k].astype('uint16'))
                            << (bw+bx).astype('uint16'), axis=[bw, bx, k]), name='z')

    Wb = tvm.decl_buffer(w.shape, w.dtype,
                         name="W",
                         offset_factor=k_i,
                         strides=[tvm.var('ldw'), tvm.var('ldw'), 1])
    Xb = tvm.decl_buffer(x.shape, x.dtype,
                         name="X",
                         offset_factor=k_i,
                         strides=[tvm.var('ldw'), 1])

    def _intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]
        vpadd = "llvm.arm.neon.vpadd.v8u8"
        vpadalu = "llvm.arm.neon.vpadalu.v16u8.v8u16"
        args_1 = tvm.const(1, 'uint32')
        args_2 = tvm.const(2, 'uint32')

        def _instr(index):
            irb = tvm.ir_builder.create()
            if index == 1:
                irb.emit(zz.vstore(0, tvm.const(0, 'uint16x8')))
                return irb.get()

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
                        for i in range(m//2):
                            cnts4[i] = tvm.call_llvm_intrin('uint8x8', vpadd,
                                                            args_1, cnts8[i*2], cnts8[i*2+1])
                        for i in range(m//4):
                            cnts2[i] = tvm.call_llvm_intrin('uint8x8', vpadd,
                                                            args_1, cnts4[i*2], cnts4[i*2+1])
                        cnts = tvm.call_pure_intrin('uint8x16', 'vectorcombine', cnts2[0], cnts2[1])
                        shifted_cnts = cnts << tvm.const(bw+bx, dtype)
                        out = tvm.call_llvm_intrin('uint16x8', vpadalu,
                                                   args_2, zz.vload(0, 'uint16x8'), shifted_cnts)
                    else: # ki == 8
                        for i in range(m):
                            ands = ww.vload([bw, i, 0], 'uint8x8') & xx.vload([bx, 0], 'uint8x8')
                            cnts8[i] = tvm.popcount(ands)
                        for i in range(m//2):
                            cnts4[i] = tvm.call_llvm_intrin('uint8x8', vpadd,
                                                            args_1, cnts8[i*2], cnts8[i*2+1])
                        for i in range(m//4):
                            cnts2[i] = tvm.call_llvm_intrin('uint8x8', vpadd,
                                                            args_1, cnts4[i*2], cnts4[i*2+1])
                        cnts = tvm.call_pure_intrin('uint8x16', 'vectorcombine', cnts2[0], cnts2[1])
                        shifted_cnts = cnts << tvm.const(bw+bx, dtype)
                        out = tvm.call_llvm_intrin('uint16x8', vpadalu,
                                                   args_2, zz.vload(0, 'uint16x8'), shifted_cnts)
                    irb.emit(zz.vstore(0, out))
            return irb.get()
        # body, reset, update
        return _instr(0), _instr(1), _instr(2)
    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(z.op, _intrin_func, binds={w: Wb, x:Xb})

# ARM specific schedule that using custom microkernel
def _schedule_spatial_conv2d_nhwc(s, data, data_q, data_pad, data_vec,
                                  kernel, kernel_q, kernel_vec,
                                  conv_out, output, last):
    # no stride and padding info here
    _, H, W, IB, CI = data_q.shape
    KH, KW, KB, _, CO = kernel_q.shape
    KB = get_const_int(KB)
    IB = get_const_int(IB)

    if data_pad is None:
        padding = (0, 0)
        _, in_h, in_w, _, _ = data_q.shape
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

    pc = _intrin_popcount(8, kfactor, KB, IB)
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

@generic.schedule_bitserial_conv2d_nhwc.register(["arm_cpu"])
def schedule_bitserial_conv2d_nhwc(outs):
    """Raspverry pi schedule for bitserial conv2d"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'spatial_bitserial_conv_nhwc' in op.tag:
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
        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

"""Convolution schedule on raspberry pi"""
from __future__ import absolute_import as _abs
import tvm


def spatial_schedule(s, conv_out, wkl, sch, outs):
    H, W = wkl.height, wkl.width
    CI = wkl.in_filter
    CO = wkl.out_filter
    HK, WK = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    HCAT, WCAT = HK-1, WK-1
    DOPAD = (HPAD != 0 and WPAD != 0)

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    UNROLL = sch.unroll

    output_vec = conv_out
    conv = output_vec.op.input_tensors[0]
    kernel_vec = conv.op.input_tensors[1]
    kernel = kernel_vec.op.input_tensors[0]
    data_vec = conv.op.input_tensors[0]
    if DOPAD:
        data_pad = data_vec.op.input_tensors[0]
        data = data_pad.op.input_tensors[0]
    else:
        data_pad = None
        data = data_vec.op.input_tensors[0]
    last = outs[0]

    A, B, C = data, kernel, last
    A0, A1 = data_pad, data_vec
    B0 = kernel_vec
    C0, C1 = conv, output_vec

    CC = s.cache_write(C0, "global")

    _, co, oh, ow, vh, vw, vc = s[C0].op.axis
    if UNROLL:
        s[C0].unroll(vw)
    s[C0].vectorize(vc)

    s[CC].compute_at(s[C0], ow)
    _, co, oh, ow, vh, vw, vc = s[CC].op.axis
    ci, dh, dw = s[CC].op.reduce_axis
    s[CC].reorder(ci, dh, vh, dw, vw, vc)

    if UNROLL:
        s[CC].unroll(vw)
    s[CC].vectorize(vc)

    ##### Schedule A
    if DOPAD:
        s[A0].compute_inline()

    _, h, _, _, _, _ = s[A1].op.axis
    if sch.ba == 1:
        oaxis = h
        paxis = h
    else:
        oh, ih = s[A1].split(h, sch.ba)
        oaxis = oh
        paxis = ih

    s[A1].parallel(paxis)
    s[A1].pragma(oaxis, "parallel_launch_point")
    s[A1].pragma(paxis, "parallel_stride_pattern")
    s[A1].pragma(oaxis, "parallel_barrier_when_finish")


    ##### Schedule B
    co, _, _, _, _ = s[B0].op.axis
    if sch.bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[B0].split(co, sch.bc)
        oaxis = oco
        paxis = ico

    s[B0].parallel(paxis)
    s[B0].pragma(oaxis, "parallel_launch_point")
    s[B0].pragma(paxis, "parallel_stride_pattern")
    s[B0].pragma(oaxis, "parallel_barrier_when_finish")


    ##### Schedule C
    n, co, h, w = s[C].op.axis
    co, vc = s[C].split(co, VC)
    oh, ow, vh, vw = s[C].tile(h, w, VH, VW)
    s[C].reorder(n, co, oh, ow, vh, vw, vc)
    if C != C1:
        s[C1].compute_inline()
    s[C0].compute_at(s[C], ow)

    if sch.bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[C].split(co, sch.bc)
        oaxis = oco
        paxis = ico

    s[C].parallel(paxis)
    s[C].pragma(oaxis, "parallel_launch_point")
    s[C].pragma(paxis, "parallel_stride_pattern")
    s[C].pragma(oaxis, "parallel_barrier_when_finish")

    return s


def im2col_schedule(s, conv_out, wkl, sch, outs):
    H, W = wkl.height, wkl.width
    CI = wkl.in_filter
    CO = wkl.out_filter
    HK, WK = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    HCAT, WCAT = HK-1, WK-1
    DOPAD = (HPAD != 0 and WPAD != 0)

    P = sch.vp
    Q = sch.vq
    UNROLL = sch.unroll

    output_vec = conv_out
    conv = output_vec.op.input_tensors[0]
    kernel_vec = conv.op.input_tensors[1]
    kernel = kernel_vec.op.input_tensors[0]
    data_vec = conv.op.input_tensors[0]
    data_col = data_vec.op.input_tensors[0]
    if DOPAD:
        data_pad = data_col.op.input_tensors[0]
        data = data_pad.op.input_tensors[0]
    else:
        data_pad = None
        data = data_col.op.input_tensors[0]
    last = outs[0]

    A, B, C = data, kernel, last
    A0, A1, A2 = data_pad, data_col, data_vec
    B0 = kernel_vec
    C0, C1 = conv, output_vec

    CC = s.cache_write(C0, "global")
    AA = s.cache_read(A2, "global", [CC])
    BB = s.cache_read(B0, "global", [CC])


    ##### Schedule CC
    _, co, im, vim, vco = s[C0].op.axis
    s[C0].unroll(vim)
    s[C0].vectorize(vco)

    s[CC].compute_at(s[C0], im)
    _, co, im, vim, vco = s[CC].op.axis
    ci, hk, wk = s[CC].op.reduce_axis
    s[CC].reorder(ci, hk, wk, vim, vco)
    s[CC].unroll(vim)
    s[CC].vectorize(vco)
    # s[CC].unroll(ccr)

    ### Schedule C
    _, co, h, w = s[C].op.axis
    im = s[C].fuse(h, w)
    im, vim = s[C].split(im, P)
    co, vco = s[C].split(co, Q)
    s[C].reorder(co, im, vim, vco)

    if sch.bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[C].split(co, sch.bc)
        oaxis = oco
        paxis = ico

    s[C].parallel(paxis)
    s[C].pragma(oaxis, "parallel_launch_point")
    s[C].pragma(paxis, "parallel_stride_pattern")
    s[C].pragma(oaxis, "parallel_barrier_when_finish")
    if C1 != C:
        s[C1].compute_inline()

    s[C0].compute_at(s[C], paxis)

    ##### Schedule A
    if DOPAD:
        s[A0].compute_inline()
    s[A1].compute_inline()
    s[AA].compute_at(s[CC], wk)
    s[AA].unroll(AA.op.axis[4])

    _, im, _, _, _, _ = s[A2].op.axis
    if sch.ba == 1:
        oaxis = im
        paxis = im
    else:
        oim, iim = s[A2].split(im, sch.ba)
        oaxis = oim
        paxis = iim

    s[A2].parallel(paxis)
    s[A2].pragma(oaxis, "parallel_launch_point")
    s[A2].pragma(paxis, "parallel_stride_pattern")
    s[A2].pragma(oaxis, "parallel_barrier_when_finish")


    ##### Schedule B
    s[BB].compute_at(s[CC], wk)
    s[BB].vectorize(BB.op.axis[4])

    co, _, _, _, _ = s[B0].op.axis
    if sch.bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[B0].split(co, sch.bc)
        oaxis = oco
        paxis = ico

    s[B0].parallel(paxis)
    s[B0].pragma(oaxis, "parallel_launch_point")
    s[B0].pragma(paxis, "parallel_stride_pattern")
    s[B0].pragma(oaxis, "parallel_barrier_when_finish")

    return s

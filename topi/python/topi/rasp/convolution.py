"""Convolution schedule on raspberry pi"""
from __future__ import absolute_import as _abs
import tvm
from collections import namedtuple

workload_entity = ['height', 'width', 'in_filter', 'out_filter',
            'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride']
Workload = namedtuple('Workload', workload_entity)

schedule1_entity = ['vh', 'vw', 'vc', 'ba', 'bc', 'unroll']
Schedule1 = namedtuple('Schedule1', schedule1_entity)

schedule2_entity = ['vp', 'vq', 'ba', 'bc', 'unroll']
Schedule2 = namedtuple('Schedule2', schedule2_entity)

# workloads of resnet18 on imagenet
# workloads of resnet18 on imagenet
workloads = [
    Workload(224, 224,   3,  64, 7, 7, 3, 3, 2, 2),
    Workload( 56,  56,  64,  64, 3, 3, 1, 1, 1, 1),
    Workload( 56,  56,  64,  64, 1, 1, 0, 0, 1, 1),
    Workload( 56,  56,  64, 128, 3, 3, 1, 1, 2, 2),
    Workload( 56,  56,  64, 128, 1, 1, 0, 0, 2, 2),
    Workload( 28,  28, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload( 28,  28, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload( 28,  28, 128, 256, 1, 1, 0, 0, 2, 2),
    Workload( 14,  14, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload( 14,  14, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload( 14,  14, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload(  7,   7, 512, 512, 3, 3, 1, 1, 1, 1),
]

spatial_schedules = [
    Schedule1( 1,  8,  4,  1,  4,  True),
    Schedule1( 1,  7,  4,  2,  4,  True),
    Schedule1( 1,  4,  8,  4,  1,  True),
    Schedule1( 1,  4,  4,  1, 16, False),
    Schedule1( 1,  4,  8,  4,  8, False),
    Schedule1( 1,  7,  4,  3,  8,  True),
    Schedule1( 1,  2,  8,  1,  8,  True),
    Schedule1( 2,  1, 16,  1,  4,  True),
    Schedule1( 1,  7,  4,  1,  1,  True),
    Schedule1( 1,  1,  8,  4, 16, False),
    Schedule1( 1,  1, 16,  1,  8, False),
    Schedule1( 1,  1,  4,  1, 16,  True),
]


def get_workload(data, kernel, stride, padding):
    _, CI, H, W = map(lambda x: x.value, data.shape)
    CO, _, HK, WK = map(lambda x: x.value, kernel.shape)
    if isinstance(padding, (list, tuple)):
        HPAD, WPAD = padding
    else:
        HPAD, WPAD = padding, padding
    if isinstance(stride, (list, tuple)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    return Workload(H, W, CI, CO, HK, WK, HPAD, WPAD, HSTR, WSTR)

def get_schedule(wkl):
    if wkl not in workloads:
        raise ValueError, "no schedule for such workload: {}".format(wkl)
    idx = workloads.index(wkl)
    return spatial_schedules[idx]

def infer_pad(data, data_pad):
    if data_pad is None:
        return 0, 0
    else:
        _, _, H, W = map(lambda x: x.value, data.shape)
        _, _, TH, TW = map(lambda x: x.value, data_pad.shape)
        hpad = (TH - H) / 2
        wpad = (TW - W) / 2
        return hpad, wpad

def infer_stride(data, kernel, out):
    _, _, IH, IW = map(lambda x: x.value, data.shape)
    _, _, KH, KW = map(lambda x: x.value, kernel.shape)
    _, _, OH, OW = map(lambda x: x.value, out.shape)
    hstride = (IH - KH) / (OH - 1)
    wstride = (IW - KW) / (OW - 1)
    return hstride, wstride

def schedule_spatial_conv2d(s, data, data_pad, data_vec,
                            kernel, kernel_vec,
                            conv_out, output, last):

    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    if data_pad is None:
        stride = infer_stride(data, kernel, output)
    else:
        stride = infer_stride(data_pad, kernel, output)
    wkl = get_workload(data, kernel, stride, padding)
    sch = get_schedule(wkl)

    H, W = wkl.height, wkl.width
    CI, CO = wkl.in_filter, wkl.out_filter
    HK, WK = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    HCAT, WCAT = HK-1, WK-1
    DOPAD = (HPAD != 0 and WPAD != 0)

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    UNROLL = sch.unroll

    A, B, C = data, kernel, last
    A0, A1 = data_pad, data_vec
    B0 = kernel_vec
    C0, C1 = conv_out, output

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


def schedule_conv2d_im2col(s, conv_out, wkl, sch, outs):
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

def schedule_convolution(outs):
    """Create schedule for tensors or return error if batch size is larager than 1"""
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if 'ewise' in op.tag or 'bcast' in op.tag:
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule conv2d
        if 'spatial_conv2d_output' in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel = kernel_vec.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            schedule_spatial_conv2d(s, data, data_pad, data_vec,
                kernel, kernel_vec, conv_out, output, outs[0])

    traverse(outs[0].op)
    return s

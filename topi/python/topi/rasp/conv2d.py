# pylint: disable=invalid-name,unused-variable,invalid-name
"""Conv2D schedule on raspberry pi"""
from __future__ import absolute_import as _abs
import tvm
from tvm import target as _target
from .. import tag
from ..nn.conv2d import conv2d, _get_schedule
from ..nn.conv2d import SpatialPack, Im2ColPack
from ..nn.conv2d import _WORKLOADS, _SCH_TO_DECL_FUNC
from ..nn.conv2d import _get_workload
from ..nn.util import infer_pad, infer_stride
from .. import generic

_SCHEDULES = [
    # float32 imagenet
    SpatialPack(1, 8, 4, 1, 4, True),
    SpatialPack(1, 7, 4, 2, 4, True),
    SpatialPack(1, 4, 8, 4, 1, True),
    SpatialPack(1, 4, 4, 1, 16, False),
    SpatialPack(1, 4, 8, 4, 8, False),
    SpatialPack(1, 7, 4, 3, 8, True),
    SpatialPack(1, 2, 8, 1, 8, True),
    SpatialPack(2, 1, 16, 1, 4, True),
    SpatialPack(1, 7, 4, 1, 1, True),
    Im2ColPack(7, 4, 1, 16, True),
    Im2ColPack(7, 4, 1, 8, False),
    Im2ColPack(7, 4, 1, 16, False),

    # float32 mobilenet
    SpatialPack(2, 2, 4, 28, 1, True),
    SpatialPack(1, 4, 8, 14, 1, False),
    SpatialPack(1, 2, 16, 8, 1, True),
    SpatialPack(1, 4, 8, 8, 8, True),
    SpatialPack(2, 2, 8, 1, 1, False),
    SpatialPack(1, 4, 8, 4, 8, False),
    SpatialPack(2, 2, 8, 1, 4, False),
    SpatialPack(2, 2, 8, 1, 8, False),
    Im2ColPack(7, 4, 1, 16, False),
    Im2ColPack(7, 4, 1, 4, True),

    # int8 imagenet
    SpatialPack(2, 2, 4, 19, 8, False),
    SpatialPack(2, 2, 8, 1, 4, True),
    SpatialPack(2, 2, 8, 7, 4, False),
    SpatialPack(2, 4, 4, 7, 16, False),
    SpatialPack(1, 7, 4, 14, 4, True),
    SpatialPack(2, 2, 8, 5, 1, False),
    SpatialPack(1, 2, 16, 3, 8, True),
    SpatialPack(1, 7, 4, 1, 16, True),
    SpatialPack(2, 2, 8, 2, 16, True),
    SpatialPack(1, 1, 8, 4, 4, True),
    SpatialPack(1, 1, 4, 1, 8, False),
    SpatialPack(1, 1, 8, 1, 16, True),

    # int8 mobilenet
    SpatialPack(2, 2, 8, 8, 1, True),
    SpatialPack(1, 7, 4, 16, 4, True),
    SpatialPack(1, 4, 8, 1, 1, True),
    SpatialPack(1, 4, 8, 1, 1, True),
    SpatialPack(1, 4, 8, 4, 8, True),
    SpatialPack(1, 4, 8, 7, 1, True),
    SpatialPack(1, 2, 8, 2, 32, True),
    SpatialPack(1, 2, 16, 2, 16, True),
    SpatialPack(1, 1, 32, 1, 16, False),
    SpatialPack(1, 1, 16, 1, 32, True),
]

@_get_schedule.register("rasp")
def _schedule_conv2d(wkl):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = _WORKLOADS.index(wkl)
    sch = _SCHEDULES[idx]
    return sch


@conv2d.register("rasp")
def _declaration_conv2d(data, kernel, stride, padding, layout, out_dtype):
    if out_dtype is None:
        out_dtype = data.dtype
    assert layout == 'NCHW', "only support NCHW convolution on rasp"
    assert data.shape[0].value == 1, "only support batch size=1 convolution on rasp"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)
    return _SCH_TO_DECL_FUNC[type(sch)](data, kernel, stride, padding, out_dtype)


def _schedule_spatial_conv2d(s, data, data_pad, data_vec,
                             kernel, kernel_vec,
                             conv_out, output, last):
    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    if data_pad is None:
        stride = infer_stride(data, kernel, output)
    else:
        stride = infer_stride(data_pad, kernel, output)
    wkl = _get_workload(data, kernel, stride, padding, output.dtype)
    sch = _get_schedule(wkl)

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

def _schedule_im2col_conv2d(s, data, data_pad, data_col, data_vec,
                            kernel, kernel_vec,
                            conv_out, output, last):
    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    if data_pad is None:
        stride = infer_stride(data, kernel, output)
    else:
        stride = infer_stride(data_pad, kernel, output)
    wkl = _get_workload(data, kernel, stride, padding, output.dtype)

    with _target.rasp():
        sch = _get_schedule(wkl)

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

    A, B, C = data, kernel, last
    A0, A1, A2 = data_pad, data_col, data_vec
    B0 = kernel_vec
    C0, C1 = conv_out, output

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

@generic.schedule_conv2d_nchw.register(["rasp"])
def schedule_conv2d(outs):
    """Create schedule for tensors"""
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

        if 'spatial_conv_output' in op.tag:
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

            _schedule_spatial_conv2d(s, data, data_pad, data_vec,
                                     kernel, kernel_vec,
                                     conv_out, output, outs[0])

        if 'im2col_conv_output' in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel = kernel_vec.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[0]
            data_col = data_vec.op.input_tensors[0]
            data = data_col.op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]
            _schedule_im2col_conv2d(s, data, data_pad, data_col, data_vec,
                                    kernel, kernel_vec,
                                    conv_out, output, outs[0])

    traverse(outs[0].op)
    return s

# pylint: disable=invalid-name,unused-variable, unused-argument
"""Schedule for depthwise_conv2d with auto fusion"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm
from .. import tag
from ..nn.util import infer_pad, infer_stride, get_pad_tuple
from .. import generic

_Workload = namedtuple('Workload',
                       ['in_dtype', 'out_dtype', 'height', 'width', 'channel', 'multiplier',
                        'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

_Schedule = namedtuple('Schedule', ['vh', 'vw', 'vc', 'bc', 'unroll'])

# workloads of depthwise conv mobile net on imagenet
_WORKLOADS = [
    _Workload('float32', 'float32', 112, 112, 32, 1, 3, 3, 1, 1, 1, 1),
    _Workload('float32', 'float32', 112, 112, 64, 1, 3, 3, 1, 1, 2, 2),
    _Workload('float32', 'float32', 56, 56, 128, 1, 3, 3, 1, 1, 1, 1),
    _Workload('float32', 'float32', 56, 56, 128, 1, 3, 3, 1, 1, 2, 2),
    _Workload('float32', 'float32', 28, 28, 256, 1, 3, 3, 1, 1, 1, 1),
    _Workload('float32', 'float32', 28, 28, 256, 1, 3, 3, 1, 1, 2, 2),
    _Workload('float32', 'float32', 14, 14, 512, 1, 3, 3, 1, 1, 1, 1),
    _Workload('float32', 'float32', 14, 14, 512, 1, 3, 3, 1, 1, 2, 2),
    _Workload('float32', 'float32', 7, 7, 1024, 1, 3, 3, 1, 1, 1, 1),
    _Workload('int16', 'int32', 112, 112, 32, 1, 3, 3, 1, 1, 1, 1),
    _Workload('int16', 'int32', 112, 112, 64, 1, 3, 3, 1, 1, 2, 2),
    _Workload('int16', 'int32', 56, 56, 128, 1, 3, 3, 1, 1, 1, 1),
    _Workload('int16', 'int32', 56, 56, 128, 1, 3, 3, 1, 1, 2, 2),
    _Workload('int16', 'int32', 28, 28, 256, 1, 3, 3, 1, 1, 1, 1),
    _Workload('int16', 'int32', 28, 28, 256, 1, 3, 3, 1, 1, 2, 2),
    _Workload('int16', 'int32', 14, 14, 512, 1, 3, 3, 1, 1, 1, 1),
    _Workload('int16', 'int32', 14, 14, 512, 1, 3, 3, 1, 1, 2, 2),
    _Workload('int16', 'int32', 7, 7, 1024, 1, 3, 3, 1, 1, 1, 1),
]

_SCHEDULES = [
    _Schedule(2, 1, 4, 1, True),
    _Schedule(2, 4, 4, 2, True),
    _Schedule(2, 1, 4, 2, False),
    _Schedule(2, 4, 4, 1, True),
    _Schedule(4, 1, 4, 8, True),
    _Schedule(1, 1, 4, 2, True),
    _Schedule(1, 1, 8, 8, True),
    _Schedule(1, 1, 4, 1, False),
    _Schedule(1, 1, 4, 4, False),
    _Schedule(2, 4, 4, 2, False),
    _Schedule(2, 7, 4, 1, True),
    _Schedule(2, 4, 4, 4, False),
    _Schedule(2, 2, 4, 4, False),
    _Schedule(2, 2, 8, 4, False),
    _Schedule(2, 2, 4, 4, True),
    _Schedule(2, 2, 8, 4, False),
    _Schedule(1, 2, 8, 4, True),
    _Schedule(1, 1, 4, 8, True),
]

def _get_workload(data, kernel, stride, padding, out_dtype):
    _, C, IH, IW = [x.value for x in data.shape]
    _, MT, KH, KW = [x.value for x in kernel.shape]
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    return _Workload(data.dtype, out_dtype, IH, IW, C, MT, KH, KW, HPAD, WPAD, HSTR, WSTR)


def _schedule(s, data, data_pad, kernel, output, last):
    padding = infer_pad(data, data_pad)
    if data_pad is None:
        stride = infer_stride(data, kernel, output)
    else:
        stride = infer_stride(data_pad, kernel, output)
    wkl = _get_workload(data, kernel, stride, padding, output.dtype)

    if wkl not in _WORKLOADS:
        return s

    # use specified schedule
    sch = _SCHEDULES[_WORKLOADS.index(wkl)]

    H, W = wkl.height, wkl.width
    CN = wkl.channel
    MT = wkl.multiplier

    HK, WK = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    VH, VW = sch.vh, sch.vw
    BC = sch.bc
    VC = sch.vc

    TH = H + 2*HPAD
    TW = W + 2*WPAD
    OH = (H + 2*HPAD - HK) / HSTR + 1
    OW = (W + 2*WPAD - WK) / WSTR + 1


    A, B, C = data, kernel, output
    A0 = data_pad

    A1 = s.cache_read(A0, "global", C)
    _, c, h, w = s[A1].op.axis
    c, vc = s[A1].split(c, VC)
    s[A1].reorder(c, h, w, vc)

    A2 = s.cache_write(A1, 'global')
    s[A0].compute_inline()
    s[A1].compute_inline()

    B0 = s.cache_read(B, "global", C)
    c, m, h, w = s[B0].op.axis
    c, vc = s[B0].split(c, VC)
    s[B0].reorder(c, m, h, w, vc)

    B1 = s.cache_write(B0, 'global')
    s[B0].compute_inline()

    _, c, h, w = s[C].op.axis
    c, vc = s[C].split(c, VC)
    s[C].reorder(c, h, w, vc)


    C0 = s.cache_write(C, 'global')
    _, c, h, w, vc = s[C0].op.axis
    dh, dw = s[C0].op.reduce_axis
    oh, ow, ih, iw = s[C0].tile(h, w, VH, VW)
    s[C0].reorder(c, oh, ow, dh, dw, ih, iw, vc)
    if sch.unroll:
        s[C0].unroll(iw)
    s[C0].vectorize(vc)


    # # s[C0].compute_at(s[C0], ow)
    launch, c, _, _ = s[C].op.axis
    s[C].pragma(launch, "parallel_launch_point")

    s[C].parallel(c)
    s[C].pragma(c, "parallel_stride_pattern")
    s[C].pragma(c, "parallel_barrier_when_finish")


    s[C0].compute_at(s[C], launch)
    _, c, h, w, vc = s[C0].op.axis
    s[C0].parallel(c)
    s[C0].pragma(c, "parallel_stride_pattern")
    s[C0].pragma(c, "parallel_barrier_when_finish")


    s[A2].compute_at(s[C0], oh)
    # parallel(s[A2], s[A2].op.axis[1], BC)

    # # s[B0].compute_at(s[C0], ow)
    s[B1].compute_at(s[C], launch)
    c, m, h, w, vc = s[B1].op.axis
    s[B1].parallel(c)
    s[B1].pragma(c, "parallel_stride_pattern")
    s[B1].pragma(c, "parallel_barrier_when_finish")

    return s


@generic.schedule_depthwise_conv2d_nchw.register(["cpu", "rasp"])
def schedule_depthwise_conv2d(outs):
    """Schedule for depthwise_conv2d nchw forward.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule depthwise_conv2d
        if op.tag == 'depthwise_conv2d_nchw':
            output = op.output(0)
            kernel = op.input_tensors[1]
            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]
            _schedule(s, data, data_pad, kernel, output, outs[0])

    traverse(outs[0].op)
    return s

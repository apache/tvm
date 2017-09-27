# pylint: disable=invalid-name
"""Schedule for depthwise_conv2d with auto fusion"""
import tvm
from ..util import get_const_tuple
from .. import tag

def _schedule(s, data, data_pad, kernel, output, last):
    A, B, C = data, kernel, output
    A0 = data_pad

    _, c, h, w = s[C].op.axis
    dh, dw = s[C].op.reduce_axis

    oh, ow, ih, iw = s[C].tile(h, w, 2, 4)
    s[C].reorder(oh, ow, dh, dw, ih, iw)
    s[C].unroll(ih)
    s[C].vectorize(iw)

    s[C].parallel(c)
    s[C].pragma(c, "parallel_launch_point")
    s[C].pragma(c, "parallel_stride_pattern")
    s[C].pragma(c, "parallel_barrier_when_finish")
    return s



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

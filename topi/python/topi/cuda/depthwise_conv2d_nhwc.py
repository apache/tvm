# pylint: disable=invalid-name
"""Schedule for depthwise_conv2d nhwc with auto fusion"""
import tvm
from ..nn.util import get_const_tuple

def schedule_depthwise_conv2d_nhwc(op):
    """Schedule for depthwise_conv2d nhwc forward ops.

    This include scale-shift and relu.

    Parameters
    ----------
    op: Operation
        The symbolic description of the operation, should be depthwise_conv2d or
        depthwise_conv2d followed by a sequence of one-to-one-mapping operators.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    s = tvm.create_schedule(op)
    def schedule_depthwise_conv2d(temp, Filter, Output):

        s[temp].compute_inline()

        FS = s.cache_read(Filter, "shared", [Output])

        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")

        b, h, w, c = s[Output].op.axis

        h_val = tvm.ir_pass.Simplify(Output.shape[1]).value
        b_val = tvm.ir_pass.Simplify(Output.shape[0]).value
        ic_val = tvm.ir_pass.Simplify(temp.shape[3]).value
        xoc, xic = s[Output].split(c, factor=ic_val)
        s[Output].reorder(xoc, b, h, w, xic)
        xo, yo, xi, yi = s[Output].tile(h, w, x_factor=2, y_factor=2)
        fused = s[Output].fuse(yo, xo)
        fused = s[Output].fuse(fused, b)
        fused = s[Output].fuse(fused, xoc)

        s[Output].bind(fused, block_x)
        s[Output].bind(xic, thread_x)

        yi, xi, ci, fi = s[FS].op.axis
        s[FS].compute_at(s[Output], fused)
        fused = s[FS].fuse(fi,ci)
        s[FS].bind(fused, thread_x)

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if OP.tag == 'ewise' or OP.tag == 'scale_shift_nhwc':
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule depthwise_conv2d
        if OP.tag == 'depthwise_conv2d_nhwc':
            PaddedInput = OP.input_tensors[0]
            Filter = OP.input_tensors[1]
            DepthwiseConv2d = OP.output(0)
            schedule_depthwise_conv2d(PaddedInput, Filter, DepthwiseConv2d)

    traverse(op)
    return s

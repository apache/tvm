# pylint: disable=invalid-name,unused-variable,invalid-name
"""Conv2D schedule on x86"""
import tvm
from .. import generic
from .. import tag

@generic.schedule_conv2d_nchw.register(["cpu"])
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

        if 'conv2d_nchw' in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            C = conv
            n, c, h, w = C.op.axis
            s[C].parallel(c)
            s[C].pragma(n, "parallel_launch_point")

    traverse(outs[0].op)
    return s

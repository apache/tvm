#pylint: disable=invalid-name, no-member, too-many-locals, too-many-statements, too-many-arguments, too-many-branches, line-too-long
"""Schedule for conv2d_nchw with auto fusion"""
import tvm
from .. import tag
from .. import generic

@generic.schedule_conv2d_nchw.register(["opengl"])
def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(conv2d, data):
        if conv2d.op in s.outputs:
            Out = conv2d
        else:
            Out = outs[0].op.output(0)
            s[conv2d].opengl()
        s[Out].opengl()
        s[data].opengl()

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].opengl()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule conv2d_nchw
        elif OP.tag.startswith('conv2d_nchw'):
            conv2d = OP.output(0)
            data = OP.input_tensors[0]
            _schedule(conv2d, data)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s

# pylint: disable=invalid-name
"""Compute definition for grouped conv2d with cuda backend"""
import tvm
from .. import generic
from .. import tag

@generic.schedule_grouped_conv2d.register(["cuda", "gpu"])
def schedule_grouped_conv2d(outs):
    """Schedule for grouped_conv2d

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of grouped_conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for grouped_conv2d
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])


    num_thread = 64
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    out = outs[0]

    def traverse(OP):
        """ inline all one-to-one-mapping operators except the last stage """
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        elif OP.tag == 'conv2d_nchw'or \
             OP.tag == 'conv2d_hwcn'or \
             OP.tag == 'conv2d_nhwc':
            ttx, _ = s[OP].split(OP.axis[1], nparts=num_thread)
            s[OP].bind(OP.axis[0], block_x)
            s[OP].bind(ttx, thread_x)
            Data = OP.input_tensors[0]
            Kernel = OP.input_tensors[1]
            traverse(Data.op)
            traverse(Kernel.op)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)
    traverse(outs[0].op)
    xto, _ = s[out].split(out.op.axis[1], nparts=num_thread)
    s[out].bind(out.op.axis[0], block_x)
    s[out].bind(xto, thread_x)

    return s

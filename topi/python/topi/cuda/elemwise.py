# pylint: disable=invalid-name, unused-variable, trailing-whitespace, no-member
"""Schedule for element wise operator"""
import tvm

def schedule_elemwise(outs):
    """Schedule for element wise op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    tvm.schedule.AutoInlineInjective(s)

    x = outs[0]
    fused = s[x].fuse(*x.op.axis)
    num_thread = 64
    bx, tx = s[x].split(fused, factor=num_thread)
    s[x].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[x].bind(tx, tvm.thread_axis("threadIdx.x"))

    return s

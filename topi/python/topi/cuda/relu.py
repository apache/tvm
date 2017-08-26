# pylint: disable=invalid-name, unused-variable, trailing-whitespace, no-member
"""Schedule for relu operator"""
import tvm

def schedule_relu(outs):
    """Schedule for relu activation op.

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
    x = outs[0]
    num_dim = len(x.shape)
    block_factor = tvm.ir_pass.Simplify(x.op.output(0).shape[num_dim-1]).value
    if block_factor % 48 == 0:
        block_factor = 48
    elif block_factor % 32 == 0:
        block_factor = 32
    bx, tx = s[x].split(x.op.axis[num_dim-1], factor=block_factor)
    for i in range(num_dim-2, 0, -1):
        bx = s[x].fuse(bx, x.op.axis[i])

    s[x].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[x].bind(tx, tvm.thread_axis("threadIdx.x"))

    tvm.schedule.AutoInlineInjective(s)

    return s

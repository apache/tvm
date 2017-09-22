# pylint: disable=invalid-name
"""Schedule for pooling operators"""
import tvm

def schedule_global_avg_pool(outs):
    """Schedule for global_avg_pool.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of global_avg_pool
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for global_avg_pool.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    temp = outs[0].op.input_tensors[0]
    Out = outs[0].op.output(0)

    s[temp].set_scope("local")

    num_thread = 8
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")

    i, c, h, w = s[Out].op.axis
    dh, dw = s[temp].op.reduce_axis
    fuse_index = s[temp].fuse(dw, dh)
    s[temp].unroll(fuse_index)
    by, ty = s[Out].split(i, factor=num_thread)
    bx, tx = s[Out].split(c, factor=num_thread)
    s[Out].reorder(by, bx, ty, tx)
    s[Out].bind(ty, thread_y)
    s[Out].bind(tx, thread_x)
    s[Out].bind(by, block_y)
    s[Out].bind(bx, block_x)

    s[temp].compute_at(s[Out], tx)

    return s

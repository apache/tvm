# pylint: disable=invalid-name, unused-variable
"""Schedule for pooling operators"""
import tvm
from .. import tag

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
    def _schedule(Pool):
        num_thread = 8
        block_x = tvm.thread_axis("blockIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
        if Pool.op in s.outputs:
            Out = Pool
            OL = s.cache_write(Pool, "local")
        else:
            Out = outs[0].op.output(0)
            s[Pool].set_scope("local")
        i, c, h, w = s[Out].op.axis
        temp = Pool.op.input_tensors[0]
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
        if Pool.op in s.outputs:
            s[OL].compute_at(s[Out], tx)
        else:
            s[Pool].compute_at(s[Out], tx)

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule global_avg_pool
        if OP.tag == 'global_avg_pool':
            Pool = OP.output(0)
            _schedule(Pool)

    traverse(outs[0].op)
    return s

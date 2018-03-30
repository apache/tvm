# pylint: disable=invalid-name, unused-variable
"""Schedule for pooling operators"""
import tvm
from .. import tag
from .. import generic

@generic.schedule_global_pool.register(["cuda", "gpu"])
def schedule_global_pool(outs):
    """Schedule for global_pool.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of global_pool
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for global_pool.
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
        by, ty = s[Out].split(i, factor=num_thread)
        bx, tx = s[Out].split(c, factor=num_thread)
        s[Out].reorder(by, bx, ty, tx)
        s[Out].bind(ty, thread_y)
        s[Out].bind(tx, thread_x)
        s[Out].bind(by, block_y)
        s[Out].bind(bx, block_x)
        if Pool.op in s.outputs:
            s[OL].compute_at(s[Out], tx)
        else:
            s[Pool].compute_at(s[Out], tx)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule global_pool
        elif OP.tag.startswith('global_pool'):
            Pool = OP.output(0)
            _schedule(Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s


@generic.schedule_pool.register(["cuda", "gpu"])
def schedule_pool(outs):
    """Schedule for pool.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of pool
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for pool.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(PaddedInput, Pool):
        s[PaddedInput].compute_inline()
        num_thread = tvm.target.current_target(allow_none=False).max_num_threads
        if Pool.op in s.outputs:
            Out = Pool
            OL = s.cache_write(Pool, "local")
        else:
            Out = outs[0].op.output(0)
            s[Pool].set_scope("local")
        fused = s[Out].fuse(*s[Out].op.axis)
        bx, tx = s[Out].split(fused, factor=num_thread)
        s[Out].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[Out].bind(tx, tvm.thread_axis("threadIdx.x"))
        if Pool.op in s.outputs:
            s[OL].compute_at(s[Out], tx)
        else:
            s[Pool].compute_at(s[Out], tx)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith('pool'):
            PaddedInput = OP.input_tensors[0]
            Pool = OP.output(0)
            _schedule(PaddedInput, Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s

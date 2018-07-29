# pylint: disable=invalid-name, unused-variable
"""Schedule for pooling operators"""
import tvm
from .. import generic
from .. import tag

def _parallel_sch(sch):
    reorder_axis = []
    if len(sch.op.axis) >= 5:
        fused = sch.fuse(sch.op.axis[0], sch.op.axis[1], sch.op.axis[2])
        reorder_axis.append(fused)
        for i in range(3, len(sch.op.axis) - 1):
            reorder_axis.append(sch.op.axis[i])
    elif len(sch.op.axis) >= 3:
        fused = sch.fuse(sch.op.axis[0], sch.op.axis[1])
        reorder_axis.append(fused)
        for i in range(2, len(sch.op.axis) - 1):
            reorder_axis.append(sch.op.axis[i])
    else:
        sch.parallel(sch.op.axis[0])
        return
    kw, kh = sch.op.reduce_axis
    fuse_k = sch.fuse(kw, kh)
    c = sch.op.axis[len(sch.op.axis) - 1]
    reorder_axis += [fuse_k, c]
    sch.reorder(*reorder_axis)
    sch.vectorize(c)
    sch.parallel(fused)


@generic.schedule_pool.register(["cpu"])
def schedule_pool(outs):
    """Schedule for pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _schedule(PaddedInput, Pool):
        if isinstance(PaddedInput.op, tvm.tensor.ComputeOp):
            s[PaddedInput].compute_inline()
        _parallel_sch(s[Pool])

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith('pool'):
            PaddedInput = OP.input_tensors[0]
            Pool = OP.output(0)
            _schedule(PaddedInput, Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


@generic.schedule_global_pool.register(["cpu"])
def schedule_global_pool(outs):
    """Schedule for global pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith('global_pool'):
            Pool = OP.output(0)
            _parallel_sch(s[Pool])
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s

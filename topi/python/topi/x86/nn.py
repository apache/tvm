# pylint: disable=invalid-name,too-many-locals
"""x86 nn operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import generic
from .. import tag

def _default_schedule(outs, auto_inline):
    """Default schedule for x86."""
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    if auto_inline:
        tvm.schedule.AutoInlineInjective(s)
        s[x].fuse(s[x].op.axis)
        return s
    if len(s[x].op.axis) == 4:
        n, c, _, _ = s[x].op.axis
        fused = s[x].fuse(n, c) # for nhwc layout, fuse n and h
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s


@generic.schedule_softmax.register(["cpu"])
def schedule_softmax(outs):
    """Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


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
    return _default_schedule(outs, False)


@generic.schedule_dense.register(["cpu"])
def schedule_dense(outs):
    """Schedule for dense

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

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'dense' in op.tag:
            C = op.output(0)
            x, y = C.op.axis

            # Write cache for blocks
            CC = s.cache_write(C, 'global')

            # Tile
            bnx = 1
            bny = 4
            _, yo, _, yi = s[C].tile(x, y, bnx, bny)
            s[CC].compute_at(s[C], yo)
            xc, yc = s[CC].op.axis
            k, = s[CC].op.reduce_axis
            ko, ki = s[CC].split(k, factor=4)
            s[CC].reorder(ko, xc, ki, yc)
            s[CC].unroll(ki)
            s[CC].vectorize(yc)

            # Vectorization
            s[C].vectorize(yi)

            # Parallelization
            s[C].parallel(yo)

    traverse(outs[0].op)
    return s

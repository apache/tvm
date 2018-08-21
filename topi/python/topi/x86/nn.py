# pylint: disable=invalid-name,too-many-locals
"""x86 nn operators"""
from __future__ import absolute_import as _abs
import tvm

from .. import generic
from ..util import traverse_inline

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
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) >= 5:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s


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

    def _callback(op):
        if 'dense' in op.tag:
            output = outs[0]
            dense = op.output(0)

            # Write cache for blocks
            if dense.op in s.outputs:
                CC = s.cache_write(dense, 'local')
            else:
                CC = dense

            # Tile
            bnx = 1
            bny = 4
            x, y = output.op.axis
            xo, yo, xi, yi = s[output].tile(x, y, bnx, bny)

            xc, yc = s[CC].op.axis
            k, = s[CC].op.reduce_axis
            ko, ki = s[CC].split(k, factor=4)
            s[CC].reorder(ko, xc, ki, yc)

            s[CC].unroll(ki)
            s[CC].vectorize(yc)

            s[output].unroll(xi)
            s[output].vectorize(yi)

            fused = s[output].fuse(xo, yo)
            s[output].parallel(fused)
            s[CC].compute_at(s[output], fused)

    traverse_inline(s, outs[0].op, _callback)
    return s

# pylint: disable=invalid-name
"""Schedule for binarization and bit-packing."""
from __future__ import absolute_import as _abs
import tvm
from .. import generic


@generic.schedule_binarize_pack.register(["cpu"])
def schedule_binarize_pack(outs):
    """Schedule for binarize_pack.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of binarize_pack
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for binarize_pack.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(Out):
        s[Out].parallel(Out.op.axis[0])

    def traverse(OP):
        # schedule binarize_pack
        if OP.tag == 'binarize_pack':
            Out = OP.output(0)
            _schedule(Out)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s

"""Multibox operations."""
from __future__ import absolute_import as _abs
from . import _make

def multibox_prior(data,
                   sizes,
                   ratios,
                   steps,
                   offsets,
                   clip):
    """Generate prior(anchor) boxes from data, sizes and ratios.

    Parameters
    ----------
    data : relay.Expr
        The input data tensor.

    sizes : tuple of float
        Tuple of sizes for anchor boxes.

    ratios : tuple of float
        Tuple of ratios for anchor boxes.

    steps : Tuple of float
        Priorbox step across y and x, -1 for auto calculation.

    offsets : tuple of int
        Priorbox center offsets, y and x respectively.

    clip : boolean
        Whether to clip out-of-boundary boxes.

    Returns
    -------
    out : relay.Expr
        3-D tensor with shape [1, h_in * w_in * (num_sizes + num_ratios - 1), 4]
    """
    return _make.multibox_prior(data, sizes, ratios, steps, offsets, clip)

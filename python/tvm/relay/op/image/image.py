"""Image operations."""
from __future__ import absolute_import as _abs
from . import _make

def resize(data,
           size,
           layout="NCHW",
           method="BILINEAR",
           align_corners=False):
    """Image resize operator.

    This operator takes data as input and does 2D scaling to the given scale factor.
    In the default case, where the data_layout is `NCHW`
    with data of shape (n, c, h, w)
    out will have a shape (n, c, size[0], size[1])

    method indicates the algorithm to be used while calculating ghe out value
    and method can be one of ("BILINEAR", "NEAREST_NEIGHBOR")

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    size: Tuple of Expr
        The out size to which the image will be resized.

    layout : str, optional
        Layout of the input.

    method : str, optional
        Scale method to used [NEAREST_NEIGHBOR, BILINEAR].

    align_corners : int, optional
        Should be true to preserve the values at the corner pixels

    Returns
    -------
    result: relay.Expr
        The resized result.
    """
    return _make.resize(data, size, layout, method, align_corners)

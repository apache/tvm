"""Yolo operations."""
from . import _make

def yolo_reorg(data, stride):
    """Yolo reorg operation used in darknet models.
    This layer shuffles the input tensor values based on the stride value.
    Along with the shuffling, it does the shape transform.
    If '(n, c, h, w)' is the data shape and 's' is stride, output shape is '(n, c*s*s, h/s, w/s)'
    Example: data(1, 4, 2, 2) = [[[[ 0  1] [ 2  3]]
                                  [[ 4  5] [ 6  7]]
                                  [[ 8  9] [10 11]]
                                  [[12 13] [14 15]]]]
             stride = 2
             ret(1, 16, 1, 1) = [[[[ 0]]  [[ 2]]  [[ 8]]  [[10]]
                                  [[ 1]]  [[ 3]]  [[ 9]]  [[11]]
                                  [[ 4]]  [[ 6]]  [[12]]  [[14]]
                                  [[ 5]]  [[ 7]]  [[13]]  [[15]]]]

    Note: stride=1 has no significance for reorg operation.

    Parameters
    ----------
    data : relay.Expr
        The input data tensor.

    stride : int
        The stride value for reorganisation.

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _make.yolo_reorg(data, stride)

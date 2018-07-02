# pylint: disable=invalid-name, unused-variable
"""
REGION Operator
====================
Region operator, used in darknet.
"""
from __future__ import absolute_import as _abs
import tvm
from ... import cpp

@tvm.target.generic_func
def region(data, num, classes, coords, background, softmax=True):
    """Region forward operators.
    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, c_in, h_in, w_in]

    num : int
        Darknet layer parameter n

    classes : int
        Darknet layer parameter classes

    coords : int
        Darknet layer parameter coords

    background : int
        Darknet layer parameter background

    softmax : boolean
        Darknet layer parameter softmax

    Returns
    -------
    out : tvm.Tensor
        4-D with shape [batch, c_in, h_in, w_in]
    """
    return cpp.yolo.region(data, num, classes, coords, background, softmax)

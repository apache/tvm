# pylint: disable=invalid-name, unused-variable
"""
YOLO Operator
=============
YOLO operator, used in darknet.
"""
from __future__ import absolute_import as _abs
import tvm
from ... import cpp

@tvm.target.generic_func
def yolo(data, num, classes):
    """YOLO forward operators.
    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, c_in, h_in, w_in]

    num : int
        Darknet layer parameter n

    classes : int
        Darknet layer parameter classes

    Returns
    -------
    out : tvm.Tensor
        4-D with shape [batch, c_in, h_in, w_in]
    """
    return cpp.yolo.yolo(data, num, classes)

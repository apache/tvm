# pylint: disable=invalid-name
"""TVM operator activation compute."""
from __future__ import absolute_import
import tvm

@tvm.tag_scope(tag='ewise_relu')
def relu(a):
    """Perform relu activation on the data

    Parameters
    ----------
    data : tvm.Tensor
        Nd Input data

    Returns
    -------
    output : tvm.Tensor
        Output with same shape
    """
    return tvm.compute(a.shape, lambda *i: \
        tvm.select((a(*i) < 0.0), 0.0, a(*i)), name='relu')

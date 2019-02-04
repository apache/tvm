"""Binary Neural Network (BNN) Operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from ..util import get_const_tuple


def batch_dot(x, y):
    assert len(x.shape) == 3 and len(y.shape) == 3, "only support 3-dim batch_dot"
    x_shape = get_const_tuple(x.shape)
    y_shape = get_const_tuple(y.shape)
    assert x_shape[0] == y_shape[0], "batch dimension doesn't match"
    assert x_shape[2] == y_shape[2], "shapes of x and y is inconsistant"
    batch, M, K = x.shape
    N = y.shape[1]
    k = tvm.reduce_axis((0, K), name='k')
    
    return tvm.compute((batch, M, N),
                       lambda b, i, j: tvm.sum(x[b, i, k] * y[b, j, k], axis=k),
                       tag='batch_dot')

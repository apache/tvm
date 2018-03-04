# pylint: disable=invalid-name, unused-variable
"""
REGION Operator
====================
Region operator, used in darknet.
"""
from __future__ import absolute_import as _abs
import tvm
from ... import transform
from ... import util
from ... import math
from ... import nn

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

    batch, c_in, h_in, w_in = util.get_const_tuple(data.shape)
    split_indices = classes+coords+1
    data_block = transform.reshape(data, (batch, num, split_indices, h_in, w_in))
    split_res = transform.split(data_block, split_indices, 2)
    split_res[0] = math.sigmoid(split_res[0])
    split_res[1] = math.sigmoid(split_res[1])
    if not background:
        split_res[coords] = math.sigmoid(split_res[coords])

    if softmax:
        offset = coords + int(not background)
        data_block_1 = []
        data_block_1.append(transform.concatenate(split_res[0:offset], 2))
        temp_out = transform.concatenate(split_res[offset:split_indices], 2)
        temp_out = nn.softmax(temp_out, axis=2)
        data_block_1.append(temp_out)
        split_res = data_block_1

    out = transform.concatenate(split_res, 2)
    out = transform.reshape(out, data.shape)
    return out

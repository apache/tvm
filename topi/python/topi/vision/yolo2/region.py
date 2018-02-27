"""
REGION Operator
====================
**Author**: `Siju Samuel <https://github.com/siju-samuel/>`_
Region operator, used in darknet.
"""
from __future__ import absolute_import as _abs
import tvm
import topi

@tvm.target.generic_func
def region(data, num, classes, coords, background, _):
    """Region forward operators.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    stride : int
        Stride value for reorganization

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    batch, c, _, _ = topi.util.get_const_tuple(data.shape)

    split_indices = c
    split_res = topi.transform.split(data, split_indices, 1)

    def channel_select_activation(range_val, coords):
        # apply activiation first 2 layers and for last coords
        if range_val == 2:
            return coords
        return range_val

    for num_it in range(num):
        for activiation_it in range(2+1): #first 2 layers + 1 coords
            index = channel_select_activation(activiation_it, coords)
            index = index + (coords + classes + 1) * num_it
            split_res[index] = topi.math.sigmoid(split_res[index])
    groups = classes + background
    batch_x = batch * num
    group_last = (groups + batch_x - 1)
    temp_out = []
    tmp_index = 0
    for index in range(split_indices):
        mode = index % (groups + batch_x)
        if mode >= batch_x:
            temp_out.insert(tmp_index, (split_res[index]))
            tmp_index = tmp_index + 1
        if mode == group_last:
            temp_out = topi.transform.concatenate(temp_out, 1)
            temp_out_x = topi.nn.softmax(temp_out, axis=1)
            temp_out = topi.transform.split(temp_out_x, groups, 1)
            for index_1 in range(groups):
                split_res[index - index_1] = temp_out[groups - index_1 - 1]
            temp_out = []
            tmp_index = 0
    out = topi.transform.concatenate(split_res, 1)
    return out

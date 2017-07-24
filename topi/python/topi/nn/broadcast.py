"""Broadcast operators"""
from __future__ import absolute_import as _abs
import tvm

def _get_bcast_info(original_shape, target_shape):
    """Get the broadcasting info.

    bcast_info = _get_bcast_info(original_shape, target_shape)
    In bcast_info:
      -1 means to the padding dim
       0 means to to be the same as the original shape
       1 means to the broadcasted dim

    E.g
    original: (2, 1, 5), target: (2, 4, 5) => bcast_info: (0, 1, 0)
    original: (2, 5), target: (4, 2, 5) => bcast_info: (-1, 0, 0)
    original: (1, 5), target: (4, 2, 5) => bcast_info: (-1, 1, 0)

    Parameters
    ----------
    original_shape : tuple of tvm.expr.IntImm
        The original shape before broadcasting
    target_shape : tuple
        The target shape

    Returns
    -------
    bcast_info : list
    """
    assert len(target_shape) >= len(original_shape)
    bcast_info = [-1 for _ in range(len(target_shape))]
    original_shape = [original_shape[i] for i in range(len(original_shape))]
    original_shape = original_shape[::-1]
    target_shape = target_shape[::-1]
    for i in range(len(original_shape)):
        if not isinstance(original_shape[i], tvm.expr.IntImm):
            raise ValueError("Element of original_shape tuple should be IntImm")
        if tvm.ir_pass.Equal(tvm.convert(target_shape[i]), original_shape[i]):
            bcast_info[i] = 0
        elif tvm.ir_pass.Equal(original_shape[i], tvm.convert(1)):
            bcast_info[i] = 1
        else:
            raise ValueError("Original Shape: {} cannot be broadcast to  {}"
                             .format(original_shape[::-1], target_shape[::-1]))
    bcast_info = bcast_info[::-1]
    return bcast_info


@tvm.tag_scope(tag="broadcast_to")
def broadcast_to(data, shape):
    """Broadcast the src to the target shape

    We follows the numpy broadcasting rule.
    See also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    Parameters
    ----------
    data : tvm.Tensor
    shape : list or tuple

    Returns
    -------
    ret : tvm.Tensor
    """
    def _bcast_to_arg_eval(data, bcast_info, *args):
        indices_tuple = []
        for i in range(len(args)):
            if bcast_info[i] == 0:
                indices_tuple.append(args[i])
            elif bcast_info[i] == 1:
                indices_tuple.append(0)
        return data[tuple(indices_tuple)]
    original_shape = data.shape
    bcast_info = _get_bcast_info(original_shape=original_shape, target_shape=shape)
    ret = tvm.compute([tvm.convert(ele) for ele in shape],
                      lambda *args: _bcast_to_arg_eval(data,
                                                       bcast_info,
                                                       *args), name=data.name + "_broadcast")
    return ret

# pylint: disable=no-member,consider-using-enumerate
"""Broadcast operators"""
from __future__ import absolute_import as _abs
import tvm
from .import tag
from .util import get_const_tuple, equal_const_int, get_const_int

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
        if equal_const_int(original_shape[i], target_shape[i]):
            bcast_info[i] = 0
        elif equal_const_int(original_shape[i], 1):
            bcast_info[i] = 1
        else:
            raise ValueError("Original Shape: {} cannot be broadcast to  {}"
                             .format(original_shape[::-1], target_shape[::-1]))
    bcast_info = bcast_info[::-1]
    return bcast_info


def _get_binary_op_bcast_shape(lhs_shape, rhs_shape):
    """Get the shape after binary broadcasting.

    We will strictly follow the broadcasting rule in numpy.

    Parameters
    ----------
    lhs_shape : tuple
    rhs_shape : tuple

    Returns
    -------
    ret_shape : tuple
    """
    ret_shape = []
    if len(lhs_shape) > len(rhs_shape):
        lhs_shape, rhs_shape = rhs_shape, lhs_shape
    for ptr in range(len(rhs_shape)):
        if ptr < len(lhs_shape):
            l_val, r_val = lhs_shape[len(lhs_shape) - 1 - ptr], \
                           rhs_shape[len(rhs_shape) - 1 - ptr]
            assert(l_val == 1 or r_val == 1 or l_val == r_val),\
                "Shape is NOT broadcastable, lhs=%s, rhs=%s"\
                %(str(lhs_shape), str(rhs_shape))
            ret_shape.append(max(l_val, r_val))
        else:
            ret_shape.append(rhs_shape[len(rhs_shape) - 1 - ptr])
    ret_shape = ret_shape[::-1]
    return ret_shape



@tvm.tag_scope(tag=tag.BROADCAST)
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
    def _bcast_to_arg_eval(data, bcast_info, *indices):
        indices_tuple = []
        for i, ind in enumerate(indices):
            if bcast_info[i] == 0:
                indices_tuple.append(ind)
            elif bcast_info[i] == 1:
                indices_tuple.append(0)
        return data[tuple(indices_tuple)]
    original_shape = data.shape
    shape = [get_const_int(i) for i in shape]
    bcast_info = _get_bcast_info(original_shape=original_shape, target_shape=shape)
    ret = tvm.compute(shape,
                      lambda *indices: _bcast_to_arg_eval(data,
                                                          bcast_info,
                                                          *indices), name=data.name + "_broadcast")
    return ret


@tvm.tag_scope(tag=tag.BROADCAST)
def broadcast_binary_op(lhs, rhs, func, name="bop"):
    """Binary operands that will automatically broadcast the inputs

    We follows the numpy broadcasting rule.
    See also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    Parameters
    ----------
    lhs : tvm.Tensor
    rhs : tvm.Tensor
    func : function

    Returns
    -------
    ret : tvm.Tensor
    """
    def _inner_arg_eval(lhs, rhs, lhs_bcast_info, rhs_bcast_info, func, *indices):
        lhs_indices = []
        rhs_indices = []
        for i, ind in enumerate(indices):
            if lhs_bcast_info[i] == 0:
                lhs_indices.append(ind)
            elif lhs_bcast_info[i] == 1:
                lhs_indices.append(0)
            if rhs_bcast_info[i] == 0:
                rhs_indices.append(ind)
            elif rhs_bcast_info[i] == 1:
                rhs_indices.append(0)
        return func(lhs[tuple(lhs_indices)], rhs[tuple(rhs_indices)])
    ret_shape = _get_binary_op_bcast_shape(get_const_tuple(lhs.shape), get_const_tuple(rhs.shape))
    lhs_bcast_info = _get_bcast_info(original_shape=lhs.shape, target_shape=ret_shape)
    rhs_bcast_info = _get_bcast_info(original_shape=rhs.shape, target_shape=ret_shape)
    ret = tvm.compute(ret_shape,
                      lambda *indices: _inner_arg_eval(lhs, rhs, lhs_bcast_info, rhs_bcast_info,
                                                       func, *indices),
                      name=lhs.name + "_" + rhs.name + "_" + name)
    return ret


def broadcast_add(lhs, rhs):
    """Binary addition with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor
    rhs : tvm.Tensor

    Returns
    -------
    ret : tvm.Tensor
    """
    return broadcast_binary_op(lhs, rhs, lambda a, b: a + b, "add")


def broadcast_mul(lhs, rhs):
    """Binary multiplication with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor
    rhs : tvm.Tensor

    Returns
    -------
    ret : tvm.Tensor
    """
    return broadcast_binary_op(lhs, rhs, lambda a, b: a * b, "mul")


def broadcast_div(lhs, rhs):
    """Binary division with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor
    rhs : tvm.Tensor

    Returns
    -------
    ret : tvm.Tensor
    """
    return broadcast_binary_op(lhs, rhs, lambda a, b: a / b, "div")


def broadcast_sub(lhs, rhs):
    """Binary subtraction with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor
    rhs : tvm.Tensor

    Returns
    -------
    ret : tvm.Tensor
    """
    return broadcast_binary_op(lhs, rhs, lambda a, b: a - b, "sub")


def broadcast_maximum(lhs, rhs):
    """Take element-wise maximum of two tensors with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor
    rhs : tvm.Tensor

    Returns
    -------
    ret : tvm.Tensor
    """
    return broadcast_binary_op(lhs, rhs, tvm.max, "maximum")


def broadcast_minimum(lhs, rhs):
    """Take element-wise minimum of two tensors with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor
    rhs : tvm.Tensor

    Returns
    -------
    ret : tvm.Tensor
    """
    return broadcast_binary_op(lhs, rhs, tvm.min, "minimum")

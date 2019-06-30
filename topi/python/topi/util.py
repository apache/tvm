# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Common topi utilities"""
from __future__ import absolute_import as _abs
from numbers import Integral

import tvm
from tvm.api import layout, bijective_layout
from . import tag

class InvalidShapeError(ValueError):
    """Invalid shape for a topi function. i.e. call winograd template for non-3x3 kernel)"""
    pass

def traverse_inline(s, final_op, callback):
    """Traverse computation graph and do auto inline

    Parameters
    ----------
    s: schedule
        The schedule
    final_op: Operation
        The final output operator.
    callback: callable
        The callback function on each op
    """
    visited = set()

    def _traverse(op):
        if op in visited:
            return
        visited.add(op)
        if tag.is_injective(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    _traverse(tensor.op)
        callback(op)

    _traverse(final_op)


def prod(x):
    """Get the product of every items in the tuple.

    Parameters
    ----------
    x: tuple
        Input tuple

    Returns
    -------
    value : Expr
        The result value
    """
    if not x:
        return tvm.const(1, "int32")
    res = x[0]
    for i in range(1, len(x)):
        res = res * x[i]
    return res


def get_const_int(expr):
    """Verifies expr is integer and get the constant value.

    Parameters
    ----------
    expr : tvm.Expr or int
        The input expression.

    Returns
    -------
    out_value : int
        The output.
    """
    if isinstance(expr, Integral):
        return expr
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        expr = tvm.ir_pass.Simplify(expr)
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        raise ValueError("Expect value to be constant int")
    return int(expr.value)


def get_const_float(expr):
    """Verifies expr is a floating point and get the constant value.

    Parameters
    ----------
    expr : tvm.Expr or float
        The input expression.

    Returns
    -------
    out_value : float
        The output.
    """
    if isinstance(expr, float):
        return float(expr)
    if not isinstance(expr, tvm.expr.FloatImm):
        expr = tvm.ir_pass.Simplify(expr)
    if not isinstance(expr, tvm.expr.FloatImm):
        raise ValueError("Expect value to be constant float")
    return float(expr.value)


def equal_const_int(expr, value):
    """Returns if expr equals value.

    Parameters
    ----------
    expr : tvm.Expr
        The input expression.

    Returns
    -------
    equal : bool
        Whether they equals.
    """
    if isinstance(expr, Integral):
        return expr == value
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        expr = tvm.ir_pass.Simplify(expr)
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        return False
    return expr.value == value


def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm, returns tuple of int.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    return tuple(get_const_int(elem) for elem in in_tuple)


def get_float_tuple(in_tuple):
    """Verifies input tuple is FloatImm, returns tuple of float.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of float
        The output.
    """
    return tuple(get_const_float(elem) for elem in in_tuple)


def simplify(expr):
    """Simplify the expression if it is Expr, directly return if it is int.

    Parameters
    ----------
    expr : Expr or int
        The input.

    Returns
    -------
    out : Expr or int
        The simplified output
    """
    return tvm.ir_pass.Simplify(expr) if isinstance(expr, tvm.expr.Expr) else expr


def ravel_index(indices, shape):
    """Flatten the index tuple to 1D

    Parameters
    ----------
    indices : tuple of int or tvm.expr.IntImm
        The input coordinates

    shape : tuple of int
        Shape of the tensor.

    Returns
    -------
    idx : int or Expr
        The index after flattening
    """
    idx = None
    for i, (shape_val, ind) in enumerate(zip(shape, indices)):
        if i != 0:
            idx = idx * shape_val + ind
        else:
            idx = ind
    return idx


def unravel_index(idx, shape):
    """Convert the flattened ind to the coordinate array

    Parameters
    ----------
    idx : int or tvm.expr.IntImm
        The 1D index

    shape : tuple of int
        Shape of the tensor

    Returns
    -------
    indices : tuple of int or tvm.expr.IntImm
        Corresponding coordinate of the 1D index
    """
    indices = []
    for i in range(len(shape) - 1, -1, -1):
        indices.append(idx % shape[i])
        idx = idx // shape[i]
    indices = indices[::-1]
    return indices


def const_matrix(matrix, name="const_matrix"):
    """convert a const numpy 2-dimensional matrix to tvm tensor

    Parameters
    ----------
    matrix: numpy.ndarray
        Const input array
    name: str, optional
        The name of output op

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    row, col = matrix.shape
    dtype = str(matrix.dtype)

    def select_array(i, j):
        now = tvm.const(0.0, dtype)
        for ii in range(row):
            for jj in range(col):
                now = tvm.expr.Select(tvm.all(i % row == ii, j % col == jj),
                                      tvm.const(matrix[ii][jj], dtype),
                                      now)
        return now

    return tvm.compute(matrix.shape, select_array, name=name)


def get_max_power2_factor(n, max_value=None):
    """Get max factor of n in power of 2. If max_value is specificed, max factor
    value will be no more max_value,

    Parameter
    ---------
    n : int
        The input value

    max_value : int, optional
        The max value for the factor

    Returns
    -------
    factor : int
        The max factor in power of 2.
    """
    x = 1
    while n % 2 == 0:
        if max_value is not None and max_value < x * 2:
            break
        x *= 2
        n /= 2
    return x


def get_shape(src_shape, src_layout, dst_layout):
    """Given a source shape, a source layout and a destination layout, infer
    the destination shape.

    Parameter
    ---------
    src_shape : tuple of int or IntImm
        Source shape

    src_layout : str or Layout
        Source layout

    dst_layout : str or Layout
        Destination layout

    Returns
    -------
    dst_shape : tuple of int
        Destination shape
    """
    if src_layout == dst_layout:
        return get_const_tuple(src_shape)

    if isinstance(src_layout, str):
        src_layout = layout(src_layout)
    if isinstance(dst_layout, str):
        dst_layout = layout(dst_layout)

    assert len(src_layout) == len(dst_layout), \
        "Incompatible layout %s vs %s" % (src_layout, dst_layout)

    layout_mapping = bijective_layout(src_layout, dst_layout)
    dst_indices = layout_mapping.forward_index(
        tvm.convert([i for i in range(len(src_layout))]))

    return get_const_tuple(tuple([src_shape[i.value] for i in dst_indices]))

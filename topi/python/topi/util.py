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
from tvm import te
from tvm.tir import layout, bijective_layout
from . import tag, cpp

class InvalidShapeError(ValueError):
    """Invalid shape for a topi function. i.e. call winograd template for non-3x3 kernel)"""

def nchw_pack_layout(layout_info):
    """Check whether the layout type is NCHWinic"""
    return layout_info[:4] == 'NCHW' and 'c' in layout_info and 'n' in layout_info

def nchw_xc_layout(layout_info):
    """Check whether the layout type is NCHWxc"""
    return layout_info[:4] == 'NCHW' and 'c' in layout_info and layout_info[4:-1].isnumeric()

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
                if isinstance(tensor.op, tvm.te.ComputeOp):
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
        return tvm.tir.const(1, "int32")
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
    if not isinstance(expr, tvm.tir.IntImm):
        expr = tvm.tir.ir_pass.Simplify(expr)
    if not isinstance(expr, tvm.tir.IntImm):
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
    if not isinstance(expr, tvm.tir.FloatImm):
        expr = tvm.tir.ir_pass.Simplify(expr)
    if not isinstance(expr, tvm.tir.FloatImm):
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
    if not isinstance(expr, tvm.tir.IntImm):
        expr = tvm.tir.ir_pass.Simplify(expr)
    if not isinstance(expr, tvm.tir.IntImm):
        return False
    return expr.value == value


def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm or Var, returns tuple of int or Var.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    ret = []
    for elem in in_tuple:
        if isinstance(elem, (tvm.tir.Var, tvm.tir.expr.Any)):
            ret.append(elem)
        elif not isinstance(elem, (tvm.tir.IntImm, int)):
            elem = tvm.tir.ir_pass.Simplify(elem)
            if not isinstance(elem, tvm.tir.IntImm):
                ret.append(elem)
            else:
                ret.append(get_const_int(elem))
        else:
            ret.append(get_const_int(elem))
    return tuple(ret)


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
    return tvm.tir.ir_pass.Simplify(expr) if isinstance(expr, tvm.tir.PrimExpr) else expr


def ravel_index(indices, shape):
    """Flatten the index tuple to 1D

    Parameters
    ----------
    indices : tuple of int or tvm.tir.IntImm
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
    idx : int or tvm.tir.IntImm
        The 1D index

    shape : tuple of int
        Shape of the tensor

    Returns
    -------
    indices : tuple of int or tvm.tir.IntImm
        Corresponding coordinate of the 1D index
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod
    indices = []
    for i in range(len(shape) - 1, -1, -1):
        indices.append(idxm(idx, shape[i]))
        idx = idxd(idx, shape[i])
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
    idxm = tvm.tir.indexmod

    def select_array(i, j):
        now = tvm.tir.const(0.0, dtype)
        for ii in range(row):
            for jj in range(col):
                now = tvm.tir.Select(tvm.tir.all(idxm(i, row) == ii, idxm(j, col) == jj),
                                     tvm.tir.const(matrix[ii][jj], dtype),
                                     now)
        return now

    return te.compute(matrix.shape, select_array, name=name)


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
        tvm.runtime.convert(list(range(len(src_layout)))))

    return get_const_tuple(tuple([src_shape[i.value] for i in dst_indices]))


def within_index(b, e, s, i):
    """Return a boolean value that indicates if i is within the given index.

    Parameter
    ---------
    b : Expr
      beginning of the index

    e : Expr
      end of the index

    s : Expr
      strides of index

    i : Expr
      array position

    Returns
    -------
    selected: Expr
        bool expression that is True is the array position would be selected
        by the index and False otherwise
    """
    bc = tvm.tir.Select(s < 0, i <= e, i < b)
    ec = tvm.tir.Select(s < 0, i > b, i >= e)
    ss = te.if_then_else(s < 0,
                         ((i - e) + (e % te.abs(s)) + 1) % te.abs(s),
                         (i - b) % s)
    return tvm.tir.Select(tvm.tir.Or(bc, ec), tvm.tir.const(False), ss.equal(0))


def make_idx(b, e, s, z, i):
    """Return the array position in the selection that corresponds to an
    array position in the full array.

    The returned value is only meaningful if within_index() returns True
    for the same set of parameters.

    Parameter
    ---------
    b : Expr
      beginning of the index

    e : Expr
      end of the index

    s : Expr
      strides of index

    z : Expr
      size of the indexed dimension

    i : Expr
      array position

    Returns
    -------
    postion: Expr
        int expression that corresponds to an array position in the selection.
    """
    bc = tvm.tir.Select(s < 0, i <= e, i < b)
    ec = tvm.tir.Select(s < 0, i > b, i >= e)

    # Clamp to array size
    b = tvm.tir.Select(z < b, z - 1, b)

    ss = tvm.tir.if_then_else(s < 0,
                              (b - i) // te.abs(s),
                              (i - b) // s)
    return tvm.tir.if_then_else(tvm.tir.Or(bc, ec), 88, ss)


def is_empty_shape(shape):
    """Check whether an input shape has dimesion with size 0.

    Parameter
    ---------
    shape : list of Expr
      Input shape

    Returns
    -------
    is_empty: bool
      Whether input shape is empty or has dimesion with size 0.
    """
    return cpp.util.is_empty_shape(shape)

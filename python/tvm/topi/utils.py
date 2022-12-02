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

import fractions
import math
from numbers import Integral
from typing import Union, Optional, Tuple

import numpy as np
import tvm
from tvm import te
from tvm.tir import Any, SizeVar, bijective_layout, layout

from . import cpp, tag


class InvalidShapeError(ValueError):
    """Invalid shape for a topi function. i.e. call winograd template for non-3x3 kernel)"""


def ncw_pack_layout(layout_info):
    """Check whether the layout type is NCWinic"""
    return layout_info[:3] == "NCW" and "c" in layout_info and "n" in layout_info


def ncw_xc_layout(layout_info):
    """Check whether the layout type is NCWxc"""
    return layout_info[:3] == "NCW" and "c" in layout_info and layout_info[3:-1].isnumeric()


def nchw_pack_layout(layout_info):
    """Check whether the layout type is NCHWinic"""
    return layout_info[:4] == "NCHW" and "c" in layout_info and "n" in layout_info


def nchw_xc_layout(layout_info):
    """Check whether the layout type is NCHWxc"""
    return layout_info[:4] == "NCHW" and "c" in layout_info and layout_info[4:-1].isnumeric()


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
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
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
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
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
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
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
    ana = None
    for elem in in_tuple:
        if isinstance(elem, (tvm.tir.Var, tvm.tir.expr.Any)):
            ret.append(elem)
        elif not isinstance(elem, (tvm.tir.IntImm, int)):
            ana = tvm.arith.Analyzer() if ana is None else ana
            elem = ana.simplify(elem)
            if not isinstance(elem, tvm.tir.IntImm):
                ret.append(elem)
            else:
                ret.append(get_const_int(elem))
        else:
            ret.append(get_const_int(elem))
    return tuple(ret)


def const_vector(vector, name="const_vector"):
    """convert a const numpy 1-dimensional vector to tvm tensor

    Parameters
    ----------
    vector: numpy.ndarray
        Const input array
    name: str, optional
        The name of output op

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)
    row = vector.shape[0]
    dtype = str(vector.dtype)
    idxm = tvm.tir.indexmod

    def select_array(i):
        now = tvm.tir.const(0.0, dtype)
        for ii in range(row):
            now = tvm.tir.Select(
                tvm.tir.all(idxm(i, row) == ii), tvm.tir.const(vector[ii], dtype), now
            )
        return now

    return te.compute(vector.shape, select_array, name=name)


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
    return tvm.arith.Analyzer().simplify(expr) if isinstance(expr, tvm.tir.PrimExpr) else expr


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
    for i, dim in enumerate(reversed(shape)):
        if dim == 0:
            indices.append(0)
        elif i == len(shape) - 1:
            # Assuming the index is in-bounds, the last coordinate is
            # already less than dim, and doesn't need the be remainder
            # mod dim.
            indices.append(idx)
        else:
            indices.append(idxm(idx, dim))
            idx = idxd(idx, dim)
    indices = indices[::-1]
    return indices


def const_matrix(matrix, name="const_matrix", attrs=None):
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
                now = tvm.tir.Select(
                    tvm.tir.all(idxm(i, row) == ii, idxm(j, col) == jj),
                    tvm.tir.const(matrix[ii][jj], dtype),
                    now,
                )
        return now

    if attrs is None:
        attrs = {"const_matrix": True, "schedule_rule": "None"}

    return te.compute(matrix.shape, select_array, name=name, attrs=attrs)


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

    assert len(src_layout) == len(dst_layout), f"Incompatible layout {src_layout} vs {dst_layout}"

    layout_mapping = bijective_layout(src_layout, dst_layout)
    dst_indices = layout_mapping.forward_index(tvm.runtime.convert(list(range(len(src_layout)))))

    return get_const_tuple(tuple([src_shape[i.value] for i in dst_indices]))


def within_index(b, e, s, i):
    """Return a boolean value that indicates if i is within the given index.

    Parameters
    ----------
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
    ss = te.if_then_else(s < 0, ((i - e) + (e % te.abs(s)) + 1) % te.abs(s), (i - b) % s)
    return tvm.tir.Select(tvm.tir.Or(bc, ec), tvm.tir.const(False), ss.equal(0))


def make_idx(b, e, s, z, i):
    """Return the array position in the selection that corresponds to an
    array position in the full array.

    The returned value is only meaningful if within_index() returns True
    for the same set of parameters.

    Parameters
    ----------
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
    position: Expr
        int expression that corresponds to an array position in the selection.
    """
    bc = tvm.tir.Select(s < 0, i <= e, i < b)
    ec = tvm.tir.Select(s < 0, i > b, i >= e)

    # Clamp to array size
    b = tvm.tir.Select(z < b, z - 1, b)

    ss = tvm.tir.if_then_else(s < 0, (b - i) // te.abs(s), (i - b) // s)
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
    return cpp.utils.is_empty_shape(shape)


def ceil_div(a, b):
    """Return ceil division of a by b"""
    return tvm.tir.indexdiv(a + (b - 1), b)


def swap(arr, axis):
    """swap arr[axis] and arr[-1]"""
    return arr[:axis] + [arr[-1]] + arr[axis + 1 : -1] + [arr[axis]]


def is_target(names):
    """Return True if the name of the current target is one of provided names"""
    names = [names] if isinstance(names, str) else names
    target = tvm.target.Target.current(allow_none=False)
    return any(name in target.keys for name in names)


def is_dynamic_shape(shape):
    """Checks if any part of a shape is dynamic"""
    return any([isinstance(x, (Any, SizeVar)) for x in shape])


class Fraction:
    """Utility class for representing integer ratios

    TVM's simplifier has specific handling for integer expressions,
    especially as they appear in indexing.  As the simplifier is used
    to check if an optimization is permissible (e.g. vectorized
    computations require linear buffer access), use of integer
    expressions may provide significant performance benefits.
    However, directly writing the resulting integer expression would
    be tedious in many cases, or may depend on a user-specified
    fractional value.

    The `Fraction` class is intended to allow for easier writing of
    integer expressions.  The operator overloads will attempt to
    generate the resulting `Fraction` (e.g. `Fraction(Var('x')+2, 3) *
    0.75` evaluates to `Fraction((Var('x')+2)*3, 12)`).  If the result
    cannot be expressed as a fraction, the `Fraction` will be
    converted to the appropriate `PrimExpr` type for us.
    (e.g. `Fraction(3,4) * Var('pi')` evaluates to `tir.Mul(0.75,
    Var('pi'))`).  This allows integer arguments to be converted into
    fractions where possible, and maintained as integer fractions
    while generating a TIR expression.


    Example
    -------

    When resizing an image from `original_width` to `output_width`,
    determining the location in the original space for a given output
    pixel.

    .. code-block::

        resized_x = Fraction.OrPrimExpr(resized_x)
        original_width = Fraction.OrPrimExpr(original_width)
        resized_width = Fraction.OrPrimExpr(resized_width)
        original_x = original_width / resized_width * (resized_x + 0.5) - 0.5

    If `original_width`, `resized_width`, and `resized_x` are all integer
    parameters, this will result in a `Fraction` equivalent to
    `Fraction(numerator = original_width * (2*resized_x + 1) - resized_width,
    denominator = 2*resized_width)`.  If any of the parameters cannot be
    represented as an integer fraction, the expression will instead use
    floating-point arithmetic.

    To return a `PrimExpr` after using a `Fraction`, use the
    `.astype(out_dtype)` method.  This method is implemented for both
    `PrimExpr` and `Fraction`, so the type coercion can be applied for both
    usages.

    .. code-block::

        output = original_x.astype('float32')

    To extract integer/fractional components of the expression, use the
    utility method `Fraction.split_whole_and_fractional_parts`.

    .. code-block::

        int_part, remainder = Fraction.split_whole_and_fractional_parts(original_x)

    """

    def __init__(
        self,
        numerator: Union[int, float, "Fraction", tvm.tir.PrimExpr],
        denominator: Optional[Union[int, float, "Fraction", tvm.tir.PrimExpr]] = None,
    ):
        """Initialize the Fraction

        Parameters
        ----------
        numerator: Union[int, float, Fraction, PrimExpr]

            The numerator of the fraction.  If a `float` or `tir.FloatImm` is
            passed, will attempt to convert to a ratio of integers.  If an exact
            representation is not found, will raise a `ValueError`.  Any other
            `tir.PrimExpr` with a floating-point data-type will also result in a
            `ValueError`.

        denominator: Optional[Union[int, float, Fraction, PrimExpr]]

            The denominator of the fraction.

            If a `float` or `tir.FloatImm` is passed, will attempt to convert to
            a ratio of integers.  If an exact representation is not found, will
            raise a `ValueError`.  Any other `tir.PrimExpr` with a floating-point
            data-type will also result in a `ValueError`.

            If `None`, set equal to 1.
        """

        def _normalize(value):
            if isinstance(value, Fraction):
                return value

            elif isinstance(value, int):
                return tvm.runtime.convert(value)

            elif (
                isinstance(value, (tvm.tir.PrimExpr, tvm.te.Tensor, tvm.te.TensorSlice))
                and "int" in value.dtype
            ):
                return value

            elif isinstance(value, (float, tvm.tir.FloatImm)):
                # A floating-point number may result from previous division
                # of integers.  Use python's `fractions.Fraction` class to
                # unpack into a rational number, so long it reproduces
                # identically the same floating-point number.
                as_float = float(value)
                as_fraction = fractions.Fraction(as_float).limit_denominator(1024)
                if as_fraction.numerator / as_fraction.denominator == as_float:
                    return Fraction(as_fraction.numerator, as_fraction.denominator)
                else:
                    raise ValueError(f"Could not represent value {value} as a ratio of integers")

            elif (
                isinstance(value, (tvm.tir.PrimExpr, tvm.te.Tensor, tvm.te.TensorSlice))
                and "float" in value.dtype
            ):
                # Any other floating-point expressions are forbidden.
                raise ValueError(f"Could not represent value {value} as a ratio of integers")

            else:
                raise TypeError(
                    f"Could not represent type {type(value)} (value = {value}) "
                    "as a ratio of integers"
                )

        numerator = _normalize(numerator)
        denominator = 1 if denominator is None else _normalize(denominator)

        if isinstance(numerator, Fraction) and isinstance(denominator, Fraction):
            self.numerator, self.denominator = (
                numerator.numerator * denominator * denominator,
                denominator.numerator * numerator.denominator,
            )
        elif isinstance(numerator, Fraction):
            self.numerator, self.denominator = (
                numerator.numerator,
                denominator * numerator.denominator,
            )
        elif isinstance(denominator, Fraction):
            self.numerator, self.denominator = (
                numerator * denominator.denominator,
                denominator.numerator,
            )
        else:
            self.numerator, self.denominator = (numerator, denominator)

        if not isinstance(self.denominator, tvm.tir.PrimExpr):
            self.denominator = tvm.tir.IntImm(self.numerator.dtype, self.denominator)

        if isinstance(self.denominator, tvm.tir.IntImm):
            assert self.denominator.value != 0

    def __repr__(self):
        return f"Fraction({self.numerator}, {self.denominator})"

    @classmethod
    def OrPrimExpr(
        cls, value: Union[int, float, "Fraction", tvm.tir.PrimExpr]
    ) -> Union[tvm.tir.PrimExpr, "Fraction"]:
        """Attempt to generate an integer fraction, with fallback to PrimExpr

        Parameters
        ----------
        value: Union[int, float, Fraction, PrimExpr]

            The value to be expressed as a fraction, if possible.

        Returns
        -------
        fraction_or_primexpr: Union[PrimExpr, Fraction]

            The resulting fraction if the value can be expressed as an
            integer fraction, otherwise the original value.  See
            docstring of `Fraction` for the allowed conversions.
        """

        try:
            return cls(value)
        except ValueError:
            return tvm.runtime.convert(value)

    @classmethod
    def split_whole_and_fractional_parts(
        cls, expr: Union["Fraction", tvm.tir.PrimExpr]
    ) -> Tuple[tvm.tir.PrimExpr, Union["Fraction", tvm.tir.PrimExpr]]:
        """Split the fraction into integer and fractional components

        Parameters
        ----------

        expr: Union[Fraction, PrimExpr]

            The expression to be split

        Returns
        -------
        int_part: PrimExpr

            The integer part of the fraction.  This is determined
            either with integer `tir.floordiv` for a `Fraction`, or
            with `tir.floor` for a `PrimExpr`.

        fractional_part: Union[PrimExpr, Fraction]

            The remaining fractional part of the initial fraction.
            This is determined either with integer `tir.floormod` for
            a `Fraction`, or by subtracting `int_part` for a
            `PrimExpr`.
        """
        if isinstance(expr, cls):
            return (expr.int_part(), expr.fractional_part())
        else:
            int_part = tvm.tir.floor(expr).astype("int32")
            return int_part, expr - int_part

    def simplify(self, analyzer: Optional["tvm.arith.Analyzer"] = None) -> "Fraction":
        """Simplify the fraction

        Parameters
        ----------
        analyzer: Optional[arith.Analyzer]

            The analyzer to use for simplification.  If None,
            construct a temporary analyzer.

        Returns
        -------
        simplified: Fraction

            The simplified fraction
        """
        if analyzer is None:
            analyzer = tvm.arith.Analyzer()
        numerator = analyzer.simplify(self.numerator)
        denominator = analyzer.simplify(self.denominator)
        if numerator == 0:
            return Fraction(0, 1)

        def _extract_coef(val):
            if isinstance(val, (int, tvm.tir.IntImm)):
                return int(val)
            elif isinstance(val, tvm.tir.Mul) and isinstance(val.b, tvm.tir.IntImm):
                return int(val.b)
            else:
                return 1

        gcd = math.gcd(_extract_coef(numerator), _extract_coef(denominator))
        if gcd != 1:
            numerator = analyzer.simplify(numerator // gcd)
            denominator = analyzer.simplify(denominator // gcd)

        return Fraction(numerator, denominator)

    def astype(self, dtype: str) -> tvm.tir.PrimExpr:
        """Convert to a tvm.tir.PrimExpr of the specified type

        The name is deliberately the same as `PrimExpr.astype`, to
        allow `expr.astype(out_dtype)` to be valid for both
        `tvm.tir.PrimExpr` and `Fraction` expressions.

        Parameters
        ----------
        dtype: str

            The TVM datatype to return.

        Returns
        -------
        value: PrimExpr

            The resulting PrimExpr

        """
        if "int" in dtype:
            return self.int_part().astype(dtype)
        else:
            frac = self.simplify()
            return frac.numerator.astype(dtype) / frac.denominator.astype(dtype)

    def int_part(self) -> tvm.tir.PrimExpr:
        """The integer part of the fraction

        Returns
        -------
        int_part: PrimExpr

            The integer part of the fraction
        """
        return tvm.tir.floordiv(self.numerator, self.denominator)

    def fractional_part(self) -> "Fraction":
        """The remainder of the fraction

        Returns
        -------
        fractional_part: Fraction

            The remainder of the fraction
        """
        return Fraction(tvm.tir.floormod(self.numerator, self.denominator), self.denominator)

    def __neg__(self):
        # Disabling the pylint check, since pylint doesn't track the
        # __init__ type annotations to determine that self.numerator
        # may not be None.

        return Fraction(
            -self.numerator,  # pylint: disable=invalid-unary-operand-type
            self.denominator,
        )

    def __mul__(self, other):
        try:
            other = Fraction(other)
        except ValueError:
            return self.astype(other.dtype) * other
        else:
            return Fraction(self.numerator * other.numerator, self.denominator * other.denominator)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = Fraction(other)
        return Fraction(self.numerator * other.denominator, self.denominator * other.numerator)

    @staticmethod
    def _with_common_denominator(lhs, rhs):
        if not isinstance(lhs.denominator, (int, tvm.tir.IntImm)) or not isinstance(
            rhs.denominator, (int, tvm.tir.IntImm)
        ):
            denom = lhs.denominator * rhs.denominator
            return Fraction(lhs.numerator * rhs.denominator, denom), Fraction(
                rhs.numerator * lhs.denominator, denom
            )

        gcd = math.gcd(int(lhs.denominator), int(rhs.denominator))
        lcm = (int(lhs.denominator) * int(rhs.denominator)) // gcd
        return Fraction(lhs.numerator * (lcm // lhs.denominator), lcm), Fraction(
            rhs.numerator * (lcm // rhs.denominator), lcm
        )

    def __add__(self, other):
        other = Fraction(other)
        self, other = Fraction._with_common_denominator(self, other)
        return Fraction(
            self.numerator + other.numerator,
            self.denominator,
        )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = Fraction(other)
        self, other = Fraction._with_common_denominator(self, other)
        return Fraction(
            self.numerator - other.numerator,
            self.denominator,
        )

    def __rsub__(self, other):
        return Fraction(other) - self

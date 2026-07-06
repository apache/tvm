# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ruff: noqa: F401
"""The expression nodes of Relax."""

import typing
from collections.abc import Callable, Mapping
from numbers import Integral, Number, Real
from typing import Any, Optional, Union

import numpy as _np  # type: ignore
import tvm_ffi
from tvm_ffi.core import String

import tvm.ir
import tvm.relax
import tvm.runtime
from tvm import DataType

from ..ir import BaseFunc, Node, Span
from ..runtime import Scriptable
from . import _ffi_api

# It is a workaround for mypy: https://github.com/python/mypy/issues/7866#issuecomment-549454370
# This feature is not supported until python 3.10:
# https://docs.python.org/3.10/whatsnew/3.10.html#pep-613-typealias
Expr = tvm.ir.Expr
Type = tvm.ir.Type  # pylint: disable=invalid-name
GlobalVar = tvm.ir.GlobalVar


def prim_value(value: Expr | int | float, dtype: str | None = None) -> Expr:
    """Convert a Python scalar or primitive expression to ``Expr``.

    Parameters
    ----------
    value : Expr | int | float
        The value to convert.

    dtype : Optional[str]
        The dtype to use when converting Python numeric values.

    Returns
    -------
    result : Expr
        The converted primitive expression.  Existing ``Expr`` inputs are
        returned unchanged.
    """
    if tvm.ir.is_prim_expr(value):
        return value
    if isinstance(value, bool | _np.bool_):
        return tvm.tirx.IntImm(dtype or "bool", int(value))
    if isinstance(value, Integral):
        return tvm.tirx.IntImm(dtype or "int64", int(value))
    if isinstance(value, Real):
        return tvm.tirx.FloatImm(dtype or "float64", float(value))
    tvm_value = tvm_ffi.convert(value)
    if tvm.ir.is_prim_expr(tvm_value):
        return tvm_value
    raise TypeError(f"Cannot convert {value} with type {type(value)} to `Expr`")


def _relax_type_is_base_of(self: Type, derived: Type) -> bool:
    """Check if this Relax type is a base of another Relax type."""

    return _ffi_api.TypeIsBaseOf(self, derived)  # type: ignore


Type.is_base_of = _relax_type_is_base_of  # type: ignore[attr-defined]


# will be registered afterwards in python/tvm/relax/op/init.py
_op_ffi_api = None  # pylint: disable=invalid-name


def _binary_op_helper(lhs: "ExprWithOp", rhs: "ExprWithOp", op: Callable) -> "ExprWithOp":
    if not isinstance(lhs, Expr):  # type: ignore
        raise ValueError("lhs must be Expr")
    if isinstance(rhs, Expr):  # type: ignore
        return op(lhs, rhs)
    elif isinstance(rhs, Number):
        raise TypeError(f"Please convert {rhs} with `const` first")
    else:
        raise TypeError(f"type {type(rhs)} not supported")


def _binary_rhs_helper(rhs: "ExprWithOp") -> "ExprWithOp":
    if isinstance(rhs, Number):
        raise TypeError(f"Please convert {rhs} with `const` first")
    raise TypeError(f"type {type(rhs)} not supported")


class ExprWithOp(Expr, Scriptable):
    """Basetype of all relax expressions that defines op overloading."""

    def astype(self, dtype: str | DataType) -> "ExprWithOp":
        """Cast the content type of the current data to dtype.

        Parameters
        ----------
        dtype : str
            The target data type.

        Note
        ----
        This function only works for TensorType Exprs.

        Returns
        -------
        result : ExprWithOp
            The result expression.
        """
        return _op_ffi_api.astype(self, dtype)  # type: ignore

    def __neg__(self) -> "ExprWithOp":
        return _op_ffi_api.negative(self)  # type: ignore

    def __lt__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.less)  # type: ignore

    def __gt__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.greater)  # type: ignore

    def __ge__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.greater_equal)  # type: ignore

    def __le__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.less_equal)  # type: ignore

    # NOTE: Cannot override __eq__ and __ne__, which will influence object equal

    def __add__(self, other: Expr) -> "ExprWithOp":
        if isinstance(self.ty, tvm.relax.TupleType) and isinstance(other, tuple):
            return tuple([*self, *other])

        return _binary_op_helper(self, other, _op_ffi_api.add)  # type: ignore

    def __radd__(self, other: Expr) -> "ExprWithOp":
        return self.__add__(other)

    def __sub__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.subtract)  # type: ignore

    def __rsub__(self, other: Expr) -> "ExprWithOp":
        return _binary_rhs_helper(other)

    def __mul__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.multiply)  # type: ignore

    def __rmul__(self, other: Expr) -> "ExprWithOp":
        return self.__mul__(other)

    def __truediv__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.divide)  # type: ignore

    def __rtruediv__(self, other: Expr) -> "ExprWithOp":
        return _binary_rhs_helper(other)

    def __floordiv__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.floor_divide)  # type: ignore

    def __rfloordiv__(self, other: Expr) -> "ExprWithOp":
        return _binary_rhs_helper(other)

    def __mod__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.mod)  # type: ignore

    def __rmod__(self, other: Expr) -> "ExprWithOp":
        return _binary_rhs_helper(other)

    def __pow__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.power)  # type: ignore

    def __rpow__(self, other: Expr) -> "ExprWithOp":
        return _binary_rhs_helper(other)

    def __call__(self, *args: list[Expr], attrs: dict[str, Any] | None = None) -> "ExprWithOp":
        """Call the variable (if it represents a function).

        Parameters
        ----------
        args: List[Expr]
            The arguments to the call.

        attr: Optional[Dict[str, object]]
            The additional attributes to the call.

        Returns
        -------
        call: ExprWithOp
            A call taking the variable as a function.
        """
        return tvm.ir.Call(self, args, attrs=attrs)

    def __getitem__(self, index: int) -> "ExprWithOp":
        """Get the i-th element of the tuple or Expr with TupleType.

        Parameters
        ----------
        index: int
            The index of the element to be retrieved.

        Note
        ----
        This function will be overridden by Tuple and ShapeExpr

        Returns
        -------
        result: ExprWithOp
            The result expression.
        """
        try:
            return TupleGetItem(self, index)
        except RuntimeError as err:
            # For Python objects with __getitem__, but without
            # __len__, tuple unpacking is done by iterating over
            # sequential indices until IndexError is raised.
            # Therefore, convert from RuntimeError to IndexError for
            # compatibility.
            if "Index out of bounds" in err.args[0]:
                raise IndexError from err
            raise


@tvm_ffi.register_object("relax.expr.If")
class If(ExprWithOp):
    """A conditional expression in Relax.

    Parameters
    ----------
    cond: Expr
        The condition.

    true_branch: Expr
        The expression evaluated when condition is true.

    false_branch: Expr
        The expression evaluated when condition is false.

    span: Optional[Span]
        Span that points to original source code
    """

    cond: Expr
    true_branch: Expr
    false_branch: Expr
    span: Span | None

    def __init__(self, cond: Expr, true_branch: Expr, false_branch: Expr, span: Span | None = None):
        self.__init_handle_by_constructor__(
            _ffi_api.If,
            cond,
            true_branch,
            false_branch,
            span,  # type: ignore
        )


@tvm_ffi.register_object("relax.expr.Tuple")
class Tuple(ExprWithOp):
    """Tuple expression that groups several fields together.

    Parameters
    ----------
    fields : Union[List[Expr], typing.Tuple[Expr, ...]]
        The fields in the tuple.

    span: Optional[Span]
        Span that points to original source code
    """

    fields: list[Expr]
    span: Span | None

    def __init__(self, fields: list[Expr] | tuple[Expr, ...], span: Span | None = None):
        if isinstance(fields, tvm.relax.Tuple):
            fields = fields.fields
        elif isinstance(getattr(fields, "ty", None), tvm.relax.TupleType):
            fields = [*fields]

        self.__init_handle_by_constructor__(_ffi_api.Tuple, fields, span)  # type: ignore

    def __getitem__(self, index: int) -> Expr:
        if index >= len(self) or index < -len(self):
            raise IndexError("Tuple index out of range")
        return self.fields[index]

    def __len__(self) -> int:
        return len(self.fields)


@tvm_ffi.register_object("relax.expr.TupleGetItem")
class TupleGetItem(ExprWithOp):
    """Get index-th item from a tuple.

    Parameters
    ----------
    tuple_value: Expr
        The input tuple expression.

    index: int
        The index.

    span: Optional[Span]
        Span that points to original source code
    """

    tuple_value: Expr
    index: int
    span: Span | None

    def __init__(self, tuple_value: Expr, index: int, span: Span | None = None):
        self.__init_handle_by_constructor__(
            _ffi_api.TupleGetItem,
            tuple_value,
            index,
            span,  # type: ignore
        )


@tvm_ffi.register_object("relax.expr.ShapeExpr")
class ShapeExpr(ExprWithOp):
    """A shape expression which allows users to construct a shape containing Expr.

    Parameters
    ----------
    values: Union[List[Expr], typing.Tuple[Expr, ...], tvm_ffi.Array]
        The values of the shape expression.

    span: Optional[Span]
        Span that points to original source code
    """

    values: list[Expr]
    span: Span | None

    def __init__(
        self,
        values: list[Expr] | tuple[Expr, ...] | tvm_ffi.Array,
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ShapeExpr, values, span)  # type: ignore

    def __getitem__(self, index):
        if index >= len(self) or index < -len(self):
            raise IndexError("ShapeExpr index out of range")
        return self.values[index]

    def __len__(self):
        return len(self.values)


def make_shape(shape: list[Any] | tuple[Any, ...]) -> ShapeExpr:
    if isinstance(shape, list | tuple):
        return ShapeExpr(shape)
    raise TypeError(
        "make_shape expects a list or tuple of shape values, "
        f"but received type {type(shape).__name__}"
    )


@tvm_ffi.register_object("relax.expr.Constant")
class Constant(ExprWithOp):
    """Constant Tensor

    Parameters
    ----------
    data: tvm.runtime.Tensor
        The data of the constant tensor.

    ty: Optional[Type]
        The type of the constant tensor. If not specified, infer it from data.

    span: Optional[Span]
        Span that points to original source code

    Note
    ----
    Scalar constants are represented by ndim-0 constant tensors.
    """

    data: tvm.runtime.Tensor
    span: Span | None

    def __init__(
        self,
        data: tvm.runtime.Tensor,
        ty: Type | None = None,
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Constant,
            data,
            ty,
            span,  # type: ignore
        )


@tvm_ffi.register_object("relax.expr.Var")
class Var(ExprWithOp):
    """The variable class for all Relax bindings.

    Parameters
    ----------
    name_hint: str
        The name hint of the variable.

    ty: Optional[Type]
        The type annotation of the variable.

    span: Optional[Span]
        Span that points to original source code
    """

    name_hint: str
    span: Span | None

    def __init__(
        self,
        name_hint: str,
        ty: Type | None = None,
        span: Span | None = None,
    ) -> None:
        if ty is not None:
            ty = tvm.runtime.convert(ty)
            if not isinstance(ty, Type):
                raise TypeError(
                    "ty needs to be an instance of Type. "
                    "If you attempt to pass in shape, "
                    "use relax.TensorType(shape, dtype)."
                )
        self.__init_handle_by_constructor__(
            _ffi_api.Var,  # type: ignore
            name_hint,
            ty,
            span,
        )


@tvm_ffi.register_object("relax.expr.DataflowVar")
class DataflowVar(Var):
    """A sub-type of the variable node used to mark dataflow variables from
    normal visible "function local" bindings.


    Parameters
    ----------
    name_hint: str
        The name hint of the variable.

    ty: Optional[Type]
        The type annotation of the variable.

    span: Optional[Span]
        Span that points to original source code
    """

    name_hint: str
    span: Span | None

    def __init__(
        self,
        name_hint: str,
        ty: Type | None = None,
        span: Span | None = None,
    ) -> None:
        # pylint: disable=super-init-not-called
        if ty is not None:
            ty = tvm.runtime.convert(ty)
            if not isinstance(ty, Type):
                raise TypeError(
                    "ty needs to be an instance of Type. "
                    "If you attempt to pass in shape, "
                    "use relax.TensorType(shape, dtype)."
                )

        self.__init_handle_by_constructor__(_ffi_api.DataflowVar, name_hint, ty, span)  # type: ignore


@tvm_ffi.register_object("relax.expr.StringImm")
class StringImm(Expr, Scriptable):
    """Represent a string literal constant."""

    value: str
    span: Span | None

    def __init__(self, value: str, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.StringImm, value, span)  # type: ignore


@tvm_ffi.register_object("relax.expr.DataTypeImm")
class DataTypeImm(Expr, Scriptable):
    """Represent a data type constant."""

    value: DataType
    span: Span | None

    def __init__(self, value: DataType | str, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DataTypeImm, value, span)  # type: ignore


@tvm_ffi.register_object("relax.expr.Binding")
class Binding(Node, Scriptable):
    """The base class of a binding in Relax."""

    var: Var
    span: Span | None


@tvm_ffi.register_object("relax.expr.MatchCast")
class MatchCast(Binding):
    """Runtime-match the value to the type.

    This operation does runtime check, populates the un-defined symbolic shape vars
    and vars in ty in the first occurrence, and insert equality assertions in
    other cases.

    Parameters
    ----------
    var: Var
        The return variable that the match cast bind to.

    value: Expr
        The input value expression.

    ty: tvm.relax.Type
        The type to match cast to.
    """

    ty: Type
    value: Expr
    span: Span | None

    def __init__(self, var: Var, value: Expr, ty: Type, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MatchCast,
            var,
            value,
            ty,
            span,  # type: ignore
        )


@tvm_ffi.register_object("relax.expr.VarBinding")
class VarBinding(Binding):
    """Variable binding, bind he variable of the lhs with the rhs.

    Parameters
    ----------
    var: Var
        The return variable that the match cast bind to.

    value: Expr
        The input value expression.

    """

    var: Var
    value: Expr
    span: Span | None

    def __init__(self, var: Var, value: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.VarBinding, var, value, span)  # type: ignore


@tvm_ffi.register_object("relax.expr.BindingBlock")
class BindingBlock(Node, Scriptable):
    """base class of binding block, bindings inside can be impure
    (with side effect or control flow)"""

    bindings: list[Binding]
    span: Span | None

    def __init__(self, bindings: list[Binding], span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BindingBlock, bindings, span)  # type: ignore


@tvm_ffi.register_object("relax.expr.DataflowBlock")
class DataflowBlock(BindingBlock):
    """dataflow block, bindings inside are pure (no side effect and no control flow)"""

    bindings: list[Binding]
    span: Span | None

    def __init__(self, bindings: list[Binding], span: Span | None = None) -> None:
        # pylint: disable=super-init-not-called
        self.__init_handle_by_constructor__(_ffi_api.DataflowBlock, bindings, span)  # type: ignore


@tvm_ffi.register_object("relax.expr.SeqExpr")
class SeqExpr(ExprWithOp):
    """A sequence of binding blocks followed by an expression."""

    blocks: list[BindingBlock]
    body: Expr
    span: Span | None

    def __init__(self, blocks: list[BindingBlock], body: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SeqExpr, blocks, body, span)  # type: ignore


@tvm_ffi.register_object("relax.expr.Function")
class Function(BaseFunc, Scriptable):
    """A Relax function."""

    params: list[Var]
    body: Expr
    ret_ty: Type
    is_pure: bool
    attrs: tvm.ir.DictAttrs
    span: Span | None

    def __init__(
        self,
        params: list[Var],
        body: Expr,
        ret_ty: Type | None = None,
        is_pure: bool | None = True,
        attrs: tvm.ir.DictAttrs | None = None,
        span: Span | None = None,
    ) -> None:
        if attrs is None:
            attrs = tvm.ir.DictAttrs({})
        self.__init_handle_by_constructor__(
            _ffi_api.Function,
            params,
            body,
            ret_ty,
            is_pure,
            attrs,
            span,
        )  # type: ignore

    @staticmethod
    def create_empty(
        params: list[Var],
        ret_ty: Type,
        is_pure: bool | None = True,
        attrs: tvm.ir.DictAttrs | None = None,
        span: Span | None = None,
    ):
        """Construct a relax.Function but without body"""
        if attrs is None:
            attrs = tvm.ir.DictAttrs({})
        return _ffi_api.FunctionCreateEmpty(params, ret_ty, is_pure, attrs, span)  # type: ignore

    def __call__(self, *args):
        """Invoke the global function.

        Parameters
        ----------
        args: List[relax.Expr]
            Arguments.
        """
        return tvm.ir.Call(self, args, None, None)

    def bind_symbolic_vars(self, binding_map: Mapping[str | tvm.tirx.Var, Expr]) -> "Function":
        """Return a new function with updated symbolic variable

        Parameters
        ----------
        binding_map: Mapping[str | tvm.tirx.Var, Expr]

            The mapping of values to be replaced.  Keys may be either
            a `tirx.Var` or a string name of the variable.  If the
            variables are referred to by name, the name must uniquely
            identify a symbolic variable in the function.

        Returns
        -------
        func: Function

            The updated function
        """

        # Relax uses int64 for symbolic variables, but the FFI
        # converts python integers into int32.
        binding_map = {
            key: tvm.tirx.const(value, "int64") if isinstance(value, int) else value
            for key, value in binding_map.items()
        }

        return _ffi_api.FunctionBindSymbolicVars(self, binding_map)  # type: ignore

    def bind_params(
        self,
        binding_map: Mapping[
            str | Var,
            int | float | Expr | tvm.runtime.Tensor | _np.ndarray,
        ],
    ) -> "Function":
        """Return a new function with updated symbolic variable

        Parameters
        ----------
        binding_map: Mapping[
                str | Var,
                int | float | Expr | tvm.runtime.Tensor | _np.ndarray,
        ]

            The mapping of values to be replaced.

            Keys may be either a `relax.Var` or a string name of the
            Relax variable.  If the variables are referred to by name,
            the name must uniquely identify a parameter in the
            function.

            Values must be a relax expression, or a value that is
            convertible into a relax expression.  The value must be
            compatible with the variable being replaced.

        Returns
        -------
        func: Function

            The updated function
        """

        def _normalize_value(value):
            # Conversions that must occur prior to the FFI
            # conversions.
            if isinstance(value, int):
                # Relax uses int64 for symbolic variables, but the FFI
                # converts python integers into int32.
                return tvm.tirx.const(value, "int64")
            elif isinstance(value, _np.ndarray | tvm.runtime.Tensor):
                return tvm.relax.const(value)
            else:
                return value

        binding_map = {key: _normalize_value(value) for key, value in binding_map.items()}

        return _ffi_api.FunctionBindParams(self, binding_map)  # type: ignore

    def inline_functions(
        self, function_map: Mapping[str | tvm.ir.GlobalVar, "Function"]
    ) -> "Function":
        return _ffi_api.FunctionInlineFunctions(self, function_map)  # type: ignore


@tvm_ffi.register_object("relax.expr.ExternFunc")
class ExternFunc(BaseFunc, ExprWithOp):
    """extern function, which represents a PackedFunc."""

    global_symbol: String
    span: Span | None

    def __init__(
        self,
        global_symbol: String,
        ty: Type | None = None,
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ExternFunc,
            global_symbol,
            ty,
            span,  # type: ignore
        )


def extern(name: str, ty: Type | None = None, span: Span | None = None):
    """Create extern function."""
    return ExternFunc(name, ty, span)


def const(
    value: bool | int | float | _np.ndarray | tvm.runtime.Tensor, dtype: str | None = None
) -> Constant:
    """Create a constant value.

    Parameters
    ----------
    value: bool | int | float | numpy.ndarray | tvm.runtime.Tensor
        The constant value.

    dtype: Optional[str]
        The data type of the resulting constant.

    Note
    ----
    When dtype is None, we use the following rule:

    - int maps to "int32"
    - float maps to "float32"
    - bool maps to "bool"
    - other using the same default rule as numpy.
    """
    # Needed for bf16 and fp8 support (does not come with numpy)
    import ml_dtypes  # pylint: disable=unused-import,import-outside-toplevel

    if isinstance(dtype, tvm.ir.PrimType):
        dtype = dtype.dtype

    if isinstance(value, Number | (bool | list)):
        value = _np.array(value, dtype=dtype)

    if not dtype:
        # when dtype is None: int maps to "int32", float maps to "float32"
        dtype = {  # type: ignore
            _np.dtype("int64"): _np.int32,  # type: ignore
            _np.dtype("float64"): _np.float32,  # type: ignore
        }.get(
            value.dtype,
            None,  # type: ignore
        )

    if isinstance(value, _np.ndarray | _np.generic):
        if dtype is not None:
            value = value.astype(dtype)
        value = tvm.runtime.tensor(value)

    if not isinstance(value, tvm.runtime.Tensor):
        raise ValueError("value has to be scalar or Tensor")

    return Constant(value)


@tvm_ffi.register_object("relax.TEPlaceholderOp")
class TEPlaceholderOp(tvm.te.tensor.Operation):
    """The placeholder op that represents a relax expression."""


def te_tensor(
    value: Expr, tir_var_map: dict[tvm.tirx.Var, tvm.tirx.Expr], name: str = "rxplaceholder"
):
    """Create a TE tensor from relax expression, with TIR variables in the
    tensor shape substituted by the given mapping

    Parameters
    ----------
    value : Expr
        The relax expression, which is required to have TensorType.

    tir_var_map : Dict[tvm.tirx.Var, tvm.tirx.Expr]
        The mapping to substitute the TIR variables appeared in the
        shape of the input Expr.

    name : str
        The name of the created tensor.
    """
    return _ffi_api.TETensor(value, tir_var_map, name)  # type: ignore


def get_shape_of(expr: Expr) -> Expr:
    """Get shape of expr.

    Parameters
    ----------
    expr: Expr
        The input expr.

    Returns
    -------
    shape: Expr
        The shape expression

    Note
    ----
    This function requires expr to be normalized.
    The function will report an error if expr's Type is not TensorType.
    It will try to return symbolic function when possible. If the tensor do not
    have a compile-time symbolic shape, the function will then choose to return
    `Call(relax.op.shape_of, [expr])`.
    """
    return _ffi_api.GetShapeOf(expr)  # type: ignore


def _update_type(expr: Expr, ty: Type | None) -> None:
    _ffi_api.UpdateType(expr, ty)  # type: ignore

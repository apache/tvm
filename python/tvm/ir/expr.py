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
"""Common expressions data structures in the IR."""

from numbers import Number

import tvm_ffi

import tvm

from ..runtime import Object, Scriptable
from . import _ffi_api, _overload_prim_expr, _tensor_expr_overload
from .base import Node, Span


@tvm_ffi.register_object("ir.Expr")
class Expr(Node):
    """Base class of all the expressions."""

    span: Span | None
    ty: "tvm.ir.Type"


def is_prim_expr(value: object) -> bool:
    """Return whether an expression has a primitive result type."""
    return isinstance(value, Expr) and isinstance(value.ty, tvm.ir.PrimType)


def is_prim_var(value: object) -> bool:
    """Return whether a value is an ordinary variable with a primitive type."""
    return isinstance(value, Var) and type(value) is Var and is_prim_expr(value)


@tvm_ffi.register_object("ir.GlobalVar")
class GlobalVar(Expr):
    """A global variable in the IR.

    GlobalVar is used to refer to the global functions
    stored in the IRModule.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
    """

    name_hint: str

    def __init__(self, name_hint: str):
        self.__init_handle_by_constructor__(_ffi_api.GlobalVar, name_hint)

    def __call__(self, *args: Expr) -> Expr:
        """Call the global variable.

        Parameters
        ----------
        args: List[Expr]
            The arguments to the call.

        Returns
        -------
        call: Expr
            A call taking the variable as a function.
        """
        from .type import PointerType

        def is_tir_arg(x):
            return (
                isinstance(x, Number)
                or is_prim_expr(x)
                or (isinstance(x, Expr) and isinstance(x.ty, PointerType))
            )

        if args and all(is_tir_arg(x) for x in args):
            return tvm.tirx.call_tir(self, *args)

        if all(isinstance(x, Expr) for x in args):
            return Call(self, args)

        arg_types = [type(x) for x in args]
        raise RuntimeError(f"Do not know how to handle GlobalVar.__call__ for types {arg_types}")


class _ExprWithOp(Expr, Scriptable):
    """Common type-directed operator behavior for core expressions."""

    __hash__ = Expr.__hash__

    def expr_ty(self):
        """Return this expression's primitive result type."""
        if is_prim_expr(self):
            return self.ty
        raise TypeError(f"Expected a primitive-valued expression, but result type is {self.ty}")

    def __add__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__add__(self, other)
        return _tensor_expr_overload.__add__(self, other)

    def __radd__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__radd__(self, other)
        return _tensor_expr_overload.__radd__(self, other)

    def __sub__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__sub__(self, other)
        return _tensor_expr_overload.__sub__(self, other)

    def __rsub__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rsub__(self, other)
        return _tensor_expr_overload.__rsub__(self, other)

    def __mul__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__mul__(self, other)
        return _tensor_expr_overload.__mul__(self, other)

    def __rmul__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rmul__(self, other)
        return _tensor_expr_overload.__rmul__(self, other)

    def __div__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__div__(self, other)
        return _tensor_expr_overload.__div__(self, other)

    def __rdiv__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rdiv__(self, other)
        return _tensor_expr_overload.__rdiv__(self, other)

    def __truediv__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__truediv__(self, other)
        return _tensor_expr_overload.__truediv__(self, other)

    def __rtruediv__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rtruediv__(self, other)
        return _tensor_expr_overload.__rtruediv__(self, other)

    def __floordiv__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__floordiv__(self, other)
        return _tensor_expr_overload.__floordiv__(self, other)

    def __rfloordiv__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rfloordiv__(self, other)
        return _tensor_expr_overload.__rfloordiv__(self, other)

    def __mod__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__mod__(self, other)
        return _tensor_expr_overload.__mod__(self, other)

    def __rmod__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rmod__(self, other)
        return _tensor_expr_overload.__rmod__(self, other)

    def __pow__(self, other):
        if is_prim_expr(self):
            return NotImplemented
        return _tensor_expr_overload.__pow__(self, other)

    def __rpow__(self, other):
        if is_prim_expr(self):
            return NotImplemented
        return _tensor_expr_overload.__rpow__(self, other)

    def __neg__(self):
        if is_prim_expr(self):
            result = _overload_prim_expr.__neg__(self)
            if result is NotImplemented:
                raise TypeError("Primitive expression overload __neg__ is not registered")
            return result
        result = _tensor_expr_overload.__neg__(self)
        if result is NotImplemented:
            raise TypeError("Tensor expression overload negative is not registered")
        return result

    def __lshift__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__lshift__(self, other)
        return NotImplemented

    def __rlshift__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rlshift__(self, other)
        return NotImplemented

    def __rshift__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rshift__(self, other)
        return NotImplemented

    def __rrshift__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rrshift__(self, other)
        return NotImplemented

    def __and__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__and__(self, other)
        return NotImplemented

    def __rand__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rand__(self, other)
        return NotImplemented

    def __or__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__or__(self, other)
        return NotImplemented

    def __ror__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__ror__(self, other)
        return NotImplemented

    def __xor__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__xor__(self, other)
        return NotImplemented

    def __rxor__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rxor__(self, other)
        return NotImplemented

    def __invert__(self):
        if is_prim_expr(self):
            result = _overload_prim_expr.__invert__(self)
            if result is NotImplemented:
                raise TypeError("Primitive expression overload __invert__ is not registered")
            return result
        return NotImplemented

    def __lt__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__lt__(self, other)
        return _tensor_expr_overload.__lt__(self, other)

    def __le__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__le__(self, other)
        return _tensor_expr_overload.__le__(self, other)

    def __eq__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__eq__(self, other)
        return Object.__eq__(self, other)

    def __ne__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__ne__(self, other)
        return Object.__ne__(self, other)

    def __gt__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__gt__(self, other)
        return _tensor_expr_overload.__gt__(self, other)

    def __ge__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__ge__(self, other)
        return _tensor_expr_overload.__ge__(self, other)

    def __nonzero__(self):
        raise ValueError(
            "Cannot use and / or / not operator to Expr, hint: use tvm.tirx.all / "
            "tvm.tirx.any, if it is None checking, use node is not None"
        )

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other, span=None):
        result = _overload_prim_expr.equal(self, other, span)
        if result is NotImplemented:
            raise TypeError("Primitive expression overload equal is not registered")
        return result

    def astype(self, dtype, span=None):
        if is_prim_expr(self):
            result = _overload_prim_expr.astype(self, dtype, span)
            if result is NotImplemented:
                raise TypeError("Primitive expression overload astype is not registered")
            return result
        result = _tensor_expr_overload.astype(self, dtype, span)
        if result is NotImplemented:
            raise TypeError("Tensor expression overload astype is not registered")
        return result

    def __call__(self, *args, attrs=None):
        if is_prim_expr(self):
            raise TypeError("A primitive-valued expression cannot be called")
        result = _tensor_expr_overload.__call__(self, *args, attrs=attrs)
        if result is NotImplemented:
            raise TypeError("Tensor expression overload __call__ is not registered")
        return result

    def __getitem__(self, index):
        if is_prim_expr(self):
            raise TypeError("A primitive-valued expression cannot be indexed")
        result = _tensor_expr_overload.__getitem__(self, index)
        if result is NotImplemented:
            raise TypeError("Tensor expression overload __getitem__ is not registered")
        return result


@tvm_ffi.register_object("ir.Call")
class Call(_ExprWithOp):
    """Core function call node."""

    op: Expr
    args: list[Expr]
    attrs: "tvm.ir.Attrs | None"
    ty_args: list["tvm.ir.Type"]
    span: Span | None

    def __init__(
        self,
        op: Expr | str,
        args: list[Expr] | tuple[Expr, ...],
        attrs: "tvm.ir.Attrs | dict | None" = None,
        ty_args: list["tvm.ir.Type"] | tuple["tvm.ir.Type", ...] | None = None,
        span: Span | None = None,
        ret_ty: "tvm.ir.Type | str | None" = None,
    ) -> None:
        # pylint: disable=import-outside-toplevel
        from .attrs import DictAttrs
        from .op import Op
        from .type import PointerType, PrimType, Type

        if isinstance(op, str):
            op = Op.get(op)
        if attrs is not None and isinstance(attrs, dict):
            attrs = DictAttrs(attrs)
        if ret_ty is None:
            ret_ty = Type.missing()
        if isinstance(ret_ty, str) and ret_ty == "handle":
            ret_ty = PointerType(PrimType("void"))
        elif ret_ty is not None and not isinstance(ret_ty, Type):
            ret_ty = PrimType(ret_ty)
        if ty_args is None:
            ty_args = []
        self.__init_handle_by_constructor__(_ffi_api.Call, ret_ty, op, args, attrs, ty_args, span)


@tvm_ffi.register_object("ir.Var")
class Var(_ExprWithOp):
    """A canonical local variable in the IR.

    Parameters
    ----------
    name : str
        The name of the variable.

    ty : Optional[Type or str]
        The exact type of the variable.  A string denotes a primitive dtype.

    span : Optional[Span]
        Span that points to the original source code.

    """

    name: str
    span: Span | None

    def __init__(
        self,
        name: str | None = None,
        ty: "tvm.ir.Type | str | None" = None,
        span: Span | None = None,
        *,
        name_hint: str | None = None,
    ) -> None:
        if name is None:
            name = name_hint
        elif name_hint is not None:
            raise TypeError("Specify either name or name_hint, not both")
        if not isinstance(name, str):
            raise TypeError("name must be a str")

        # pylint: disable=import-outside-toplevel
        from .type import PointerType, PrimType, Type

        if isinstance(ty, str):
            ty = PointerType(PrimType("void")) if ty == "handle" else PrimType(ty)
        elif ty is not None:
            ty = tvm.runtime.convert(ty)
            if not isinstance(ty, Type):
                raise TypeError("ty must be a Type or primitive dtype string")
        self.__init_handle_by_constructor__(_ffi_api.Var, name, ty, span)


@tvm_ffi.register_object("ir.Range")
class Range(Node, Scriptable):
    """Represent a range in TVM.

    You do not need to create a Range explicitly.
    Python lists and tuples will be converted automatically to a Range in API functions.

    Parameters
    ----------
    begin : Expr
        The begin value of the range when end is None.
        Otherwise it is the length of the range.

    end : Optional[Expr]
        The end value of the range.

    span : Optional[Span]
        The location of this node in the source code.

    Note
    ----
    The constructor creates the range `[begin, end)`
    if the end argument is not None. Otherwise, it creates `[0, begin)`.
    """

    min: Expr
    extent: Expr
    span: Span | None

    def __init__(self, begin: Expr, end: Expr | None = None, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Range, begin, end, span)

    @staticmethod
    def from_min_extent(min_value: Expr, extent: Expr, span: Span | None = None) -> "Range":
        """Construct a Range by min and extent.

        This constructs a range in [min_value, min_value + extent)

        Parameters
        ----------
        min_value : Expr
            The minimum value of the range.

        extent : Expr
            The extent of the range.

        span : Optional[Span]
            The location of this node in the source code.

        Returns
        -------
        rng : Range
            The constructed range.
        """
        return _ffi_api.Range_from_min_extent(min_value, extent, span)

    def __eq__(self, other: Object) -> bool:
        return tvm_ffi.structural_equal(self, other)

    def __ne__(self, other: Object) -> bool:
        return not self.__eq__(other)

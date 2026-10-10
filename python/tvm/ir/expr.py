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

from collections.abc import Callable
from functools import partial
from inspect import getattr_static
from numbers import Number

import tvm_ffi

import tvm
from tvm.ir.base import UnknownLoc

from ..runtime import Object
from . import _ffi_api, _tensor_expr_overload
from ._constant import const
from .base import Location, Node, Scriptable

_ATTRIBUTE_MISSING = object()


def _convert_subscript_index(index):
    """Convert Python indexing syntax into an FFI subscript descriptor."""

    def convert(value):
        if value is None or is_prim_expr(value):
            return value
        return const(value)

    if isinstance(index, slice):
        return (convert(index.start), convert(index.stop), convert(index.step))
    if index is Ellipsis or index is None:
        raise TypeError("Ellipsis and newaxis are not supported in expression subscriptions")
    return convert(index)


@tvm_ffi.register_object("ir.Expr")
class Expr(Node):
    """Base class of all the expressions."""

    loc: Location
    ty: "tvm.ir.Type"

    def __getattr__(self, name):
        # Reflected fields are ordinary Python properties and have already had
        # their chance to resolve before this type-directed fallback.
        try:
            ty = object.__getattribute__(self, "ty")
        except AttributeError:
            ty = None
        if ty is not None and not name.startswith("_"):
            if name in ty.__expr_properties__:
                getter = ty.__expr_properties__[name]
                if not callable(getter):
                    raise TypeError(
                        f"Declared expression property {name!r} must have a callable getter"
                    )
                return getter(ty, self)
            if name in ty.__expr_methods__:
                method = getattr(ty, name)
                if not callable(method):
                    raise TypeError(f"Declared expression method {name!r} must be callable")
                bound = partial(method, self)
                bound.__doc__ = method.__doc__
                return bound
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        # Do not invoke getters to distinguish real attributes from the fallback.
        # Existing descriptors and instance attributes keep their usual behavior.
        if (
            not name.startswith("_")
            and getattr_static(self, name, _ATTRIBUTE_MISSING) is _ATTRIBUTE_MISSING
        ):
            try:
                ty = object.__getattribute__(self, "ty")
            except AttributeError:
                ty = None
            if ty is not None and name in ty.__expr_properties__:
                raise AttributeError(f"Expression property {name!r} is read-only")
        super().__setattr__(name, value)

    def __dir__(self):
        names = set(super().__dir__())
        try:
            ty = object.__getattribute__(self, "ty")
        except AttributeError:
            return sorted(names)
        names.update(name for name in ty.__expr_methods__ if not name.startswith("_"))
        names.update(name for name in ty.__expr_properties__ if not name.startswith("_"))
        return sorted(names)

    def __getitem__(self, index):
        if isinstance(self.ty, tvm.ir.MissingType):
            # Preserve Relax's pre-normalization tuple access: operator calls
            # have a missing result type until the block builder infers it.
            return TupleGetItem(self, index)

        indices = tuple(index) if isinstance(index, tuple | list) else (index,)
        return _ffi_api.SubscriptExprRealize(
            self, [_convert_subscript_index(item) for item in indices], UnknownLoc()
        )


@tvm_ffi.register_object("ir.StagingExpr")
class StagingExpr(Expr):
    """A traversable expression eliminated before executable IR."""


@tvm_ffi.register_object("ir.OpaqueExpr")
class OpaqueExpr(Expr):
    """Base class for opaque values that must be removed from finished IR."""


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
        return Call(self, args)


class ExprOperand:
    """Python operator surface for anything that denotes an expression."""

    __slots__ = ()
    __hash__ = object.__hash__

    def expr_ty(self):
        """Return this expression's primitive result type."""
        if is_prim_expr(self):
            return self.ty
        raise TypeError(f"Expected a primitive-valued expression, but result type is {self.ty}")

    def __add__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__add__(self, other)
        result = _tensor_expr_overload.__add__(self, other)
        return result

    def __radd__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__radd__(self, other)
        result = _tensor_expr_overload.__radd__(self, other)
        return result

    def __sub__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__sub__(self, other)
        result = _tensor_expr_overload.__sub__(self, other)
        return result

    def __rsub__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rsub__(self, other)
        result = _tensor_expr_overload.__rsub__(self, other)
        return result

    def __mul__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__mul__(self, other)
        result = _tensor_expr_overload.__mul__(self, other)
        return result

    def __rmul__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rmul__(self, other)
        result = _tensor_expr_overload.__rmul__(self, other)
        return result

    def __div__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__div__(self, other)
        result = _tensor_expr_overload.__div__(self, other)
        return result

    def __rdiv__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rdiv__(self, other)
        result = _tensor_expr_overload.__rdiv__(self, other)
        return result

    def __truediv__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__truediv__(self, other)
        result = _tensor_expr_overload.__truediv__(self, other)
        return result

    def __rtruediv__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rtruediv__(self, other)
        result = _tensor_expr_overload.__rtruediv__(self, other)
        return result

    def __floordiv__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__floordiv__(self, other)
        result = _tensor_expr_overload.__floordiv__(self, other)
        return result

    def __rfloordiv__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rfloordiv__(self, other)
        result = _tensor_expr_overload.__rfloordiv__(self, other)
        return result

    def __mod__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__mod__(self, other)
        result = _tensor_expr_overload.__mod__(self, other)
        return result

    def __rmod__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__rmod__(self, other)
        result = _tensor_expr_overload.__rmod__(self, other)
        return result

    def __pow__(self, other):
        if is_prim_expr(self):
            return NotImplemented
        result = _tensor_expr_overload.__pow__(self, other)
        return result

    def __rpow__(self, other):
        if is_prim_expr(self):
            return NotImplemented
        result = _tensor_expr_overload.__rpow__(self, other)
        return result

    def __neg__(self):
        if is_prim_expr(self):
            result = _overload_prim_expr.__neg__(self)
            if result is NotImplemented:
                raise TypeError("Primitive expression overload __neg__ is not registered")
            return result
        result = _tensor_expr_overload.__neg__(self)
        if result is NotImplemented:
            raise TypeError(f"Operator overloading is not supported for expression type {self.ty}")
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
        raise TypeError(f"Operator overloading is not supported for expression type {self.ty}")

    def __lt__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__lt__(self, other)
        result = _tensor_expr_overload.__lt__(self, other)
        return result

    def __le__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__le__(self, other)
        result = _tensor_expr_overload.__le__(self, other)
        return result

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
        result = _tensor_expr_overload.__gt__(self, other)
        return result

    def __ge__(self, other):
        if is_prim_expr(self):
            return _overload_prim_expr.__ge__(self, other)
        result = _tensor_expr_overload.__ge__(self, other)
        return result

    def __nonzero__(self):
        raise ValueError(
            "Cannot use and / or / not operator to Expr, hint: use tvm.tirx.all / "
            "tvm.tirx.any, if it is None checking, use node is not None"
        )

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other, loc=UnknownLoc()):
        if not is_prim_expr(self):
            raise TypeError(f"Operator overloading is not supported for expression type {self.ty}")
        result = _overload_prim_expr.equal(self, other, loc)
        if result is NotImplemented:
            raise TypeError("Primitive expression overload equal is not registered")
        return result

    def astype(self, dtype, loc=UnknownLoc()):
        if is_prim_expr(self):
            result = _overload_prim_expr.astype(self, dtype, loc)
            if result is NotImplemented:
                raise TypeError("Primitive expression overload astype is not registered")
            return result
        result = _tensor_expr_overload.astype(self, dtype, loc)
        if result is NotImplemented:
            raise TypeError(f"Operator overloading is not supported for expression type {self.ty}")
        return result


class _ExprCallable:
    """Function-call capability for expression operands that can denote functions."""

    __slots__ = ()

    def __call__(self, *args, attrs=None):
        if is_prim_expr(self):
            raise TypeError("A primitive-valued expression cannot be called")
        result = _tensor_expr_overload.__call__(self, *args, attrs=attrs)
        if result is NotImplemented:
            raise TypeError(f"Expression of type {self.ty} cannot be called")
        return result


class ExprWithOp(ExprOperand, Expr, Scriptable):
    """Common type-directed operator behavior for core expressions."""

    __hash__ = Expr.__hash__


class _CallableExprWithOp(_ExprCallable, ExprWithOp):
    """Common operator behavior for expression nodes that support function calls."""


@tvm_ffi.register_object("ir.Tuple")
class Tuple(_CallableExprWithOp):
    """Tuple expression that groups several fields together.

    Parameters
    ----------
    fields : list[Expr] | tuple[Expr, ...]
        The fields in the tuple.

    loc : Location
        Location that points to the original source code.
    """

    fields: list[Expr]
    loc: Location

    def __init__(self, fields: list[Expr] | tuple[Expr, ...], loc: Location = UnknownLoc()):
        if isinstance(fields, Tuple):
            fields = fields.fields
        elif isinstance(getattr(fields, "ty", None), tvm.ir.TupleType):
            fields = [*fields]

        self.__init_handle_by_constructor__(_ffi_api.Tuple, fields, loc)

    def __getitem__(self, index: int) -> Expr:
        if index >= len(self) or index < -len(self):
            raise IndexError("Tuple index out of range")
        return self.fields[index]

    def __len__(self) -> int:
        return len(self.fields)


@tvm_ffi.register_object("ir.TupleGetItem")
class TupleGetItem(_CallableExprWithOp):
    """Get the index-th item from a tuple.

    Parameters
    ----------
    tuple_value : Expr
        The input tuple expression.

    index : int
        The field index.

    loc : Location
        Location that points to the original source code.
    """

    tuple_value: Expr
    index: int
    loc: Location

    def __init__(self, tuple_value: Expr, index: int, loc: Location = UnknownLoc()):
        self.__init_handle_by_constructor__(_ffi_api.TupleGetItem, tuple_value, index, loc)


@tvm_ffi.register_object("ir.TensorLoad")
class TensorLoad(_CallableExprWithOp):
    """An indexed load from an expression source.

    TensorLoad objects are constructed by a dialect-specific helper that
    validates the source and derives the result type.
    """

    source: Expr
    indices: list[Expr]
    loc: Location

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "TensorLoad cannot be constructed directly; use a dialect-specific load helper"
        )


@tvm_ffi.register_object("ir.Constant")
class Constant(ExprWithOp):
    """Base class of literal constants."""


@tvm_ffi.register_object("ir.GenericConst")
class GenericConst(_ExprCallable, Constant):
    """A literal payload with an explicit expression type."""

    def __init__(self, value, ty: "tvm.ir.Type", loc: Location = UnknownLoc()) -> None:
        self.__init_handle_by_constructor__(_ffi_api.GenericConst, value, ty, loc)

    def __bool__(self) -> bool:
        return True


@tvm_ffi.register_object("ir.DataTypeImm")
class DataTypeImm(Constant):
    """A data type literal whose expression type is AnyType.

    Parameters
    ----------
    value : str or tvm.DataType
        The represented data type.
    loc : Location, optional
        The source location of the literal.
    """

    value: tvm.DataType

    def __init__(self, value: str | tvm.DataType, loc: Location = UnknownLoc()) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DataTypeImm, value, loc)


@tvm_ffi.register_object("ir.StringImm")
class StringImm(Constant):
    """A string literal with StringType."""

    value: str

    def __init__(self, value: str, loc: Location = UnknownLoc()) -> None:
        self.__init_handle_by_constructor__(_ffi_api.StringImm, value, loc)

    def __eq__(self, other) -> bool:
        return self.value == (other.value if isinstance(other, StringImm) else other)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    __hash__ = Expr.__hash__


@tvm_ffi.register_object("ir.Call")
class Call(_CallableExprWithOp):
    """Core function call node.

    Omitted or ``None`` ``ty`` uses available result inference, or a missing
    type when no deduction is available. Explicit types, including
    ``Type.missing()``, are preserved exactly. Inference errors propagate.
    Construction permits provisional IR; :meth:`validate` checks the operator
    contract explicitly after inputs are ready.
    """

    op: Expr
    args: list[Expr]
    attrs: "tvm.ir.Attrs | None"
    ty_args: list["tvm.ir.Type"]
    loc: Location

    def __init__(
        self,
        op: Expr | str,
        args: list[Expr] | tuple[Expr, ...],
        attrs: "tvm.ir.Attrs | dict | None" = None,
        ty_args: list["tvm.ir.Type"] | tuple["tvm.ir.Type", ...] | None = None,
        loc: Location = UnknownLoc(),
        ty: "tvm.ir.Type | str | None" = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Call, *self._normalize_constructor_args(op, args, attrs, ty_args, loc, ty)
        )

    @staticmethod
    def _normalize_constructor_args(op, args, attrs, ty_args, loc, ty):
        # pylint: disable=import-outside-toplevel
        from .attrs import DictAttrs
        from .op import Op
        from .type import PointerType, PrimType, Type

        if isinstance(op, str):
            op = Op.get(op)
        if attrs is not None and isinstance(attrs, dict):
            attrs = DictAttrs(attrs)
        if isinstance(ty, str) and ty == "handle":
            ty = PointerType(PrimType("void"))
        elif ty is not None and not isinstance(ty, Type):
            ty = PrimType(ty)
        if ty_args is None:
            ty_args = []
        return ty, op, args, attrs, ty_args, loc

    def validate(self) -> None:
        """Check the registered operator contract without changing this Call."""
        _ffi_api.CallValidate(self)


def reinfer_type(call: Call) -> "tvm.ir.Type":
    """Derive a Call's result type from its current inputs without changing the Call.

    The operator must register a fixed return type or a context-free inference
    rule. This does not invoke the operator's validator.
    """
    return _ffi_api.reinfer_type(call)


@tvm_ffi.register_object("ir.TensorRegion")
class TensorRegion(Expr, Scriptable):
    """A region of an arbitrary tensor expression.

    Parameters
    ----------
    source : Expr
        The source expression.

    region : list[Range]
        The ranges describing the region.

    ty : tvm.ir.Type
        The result type, including any dialect-specific subscript semantics.

    loc : Location
        The location of the expression in the source code.
    """

    source: Expr
    region: list["Range"]

    def __init__(
        self,
        source: Expr,
        region: list["Range"],
        ty: "tvm.ir.Type",
        loc: Location = UnknownLoc(),
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.TensorRegion, source, region, ty, loc)


@tvm_ffi.register_object("ir.Var")
class Var(_CallableExprWithOp):
    """A canonical local variable in the IR.

    Parameters
    ----------
    name : str
        The name of the variable.

    ty : Optional[Type or str]
        The exact type of the variable.  A string denotes a primitive dtype.

    loc : Location
        Location that points to the original source code.

    """

    name: str
    loc: Location

    def __init__(
        self,
        name: str | None = None,
        ty: "tvm.ir.Type | str | None" = None,
        loc: Location = UnknownLoc(),
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
        self.__init_handle_by_constructor__(_ffi_api.Var, name, ty, loc)


def _lambda_type(annotation):
    """Normalize explicit types and the script scalar constructor forms."""
    if isinstance(annotation, tvm.ir.Type):
        return annotation
    if isinstance(annotation, str):
        return Var("", annotation).ty
    if isinstance(annotation, tvm.DataType):
        return tvm.ir.PrimType(annotation)
    dtype = getattr(annotation, "_dtype_str", None)
    if dtype is not None:
        return tvm.ir.PrimType(dtype)
    if callable(annotation):
        value = annotation()
        if isinstance(value, Expr):
            return value.ty
        if isinstance(value, tvm.ir.Type):
            return value
    raise TypeError("Lambda parameter and return annotations must be explicit IR types")


def _lambda_result(value):
    if isinstance(value, tuple | list):
        return Tuple([_lambda_result(field) for field in value])
    if isinstance(value, Number):
        value = const(value)
    elif isinstance(value, str):
        value = StringImm(value)
    else:
        value = tvm.runtime.convert(value)
    if not isinstance(value, Expr):
        raise TypeError("Lambda body must be an Expr or a tuple/list of expressions")
    return value


@tvm_ffi.register_object("ir.LambdaExpr")
class LambdaExpr(StagingExpr, Scriptable):
    """A typed staging expression representing a lambda computation.

    LambdaExpr records computations such as reduction combiners and predication
    rules. Its body may describe computations on runtime values.

    Parameters are bound within the expression body, which may produce a scalar
    or tuple result. The lambda has a FuncType describing its parameter and
    return types.

    As a StagingExpr, LambdaExpr is eliminated during compilation and does not
    remain in executable IR.

    Parameters
    ----------
    parameter_types : list[Type]
        Explicit parameter types, in callable argument order. Primitive dtype
        strings and script scalar constructors are also accepted.
    function : Callable
        A callable evaluated once with one fresh typed Var per supplied type.
        Tuple/list results become shared IR Tuple expressions.
    ret_type : Type, optional
        An exact return-type check. No implicit conversion or cast is inserted.
    """

    vars: list[Var]
    body: Expr

    def __init__(self, parameter_types, function: Callable, *, ret_type=None):
        variables = [Var(f"arg{i}", _lambda_type(ty)) for i, ty in enumerate(parameter_types)]
        body = _lambda_result(function(*variables))
        if ret_type is not None:
            expected = _lambda_type(ret_type)
            if not tvm_ffi.structural_equal(expected, body.ty):
                raise TypeError("LambdaExpr return annotation does not match the body type")
        self.__init_handle_by_constructor__(_ffi_api.LambdaExpr, variables, body)

    def apply(self, arguments: list[Expr]) -> Expr:
        """Substitute arguments simultaneously for the lambda's bound variables."""
        return _ffi_api.LambdaExprApply(self, [_lambda_result(arg) for arg in arguments])


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

    loc : Location
        The location of this node in the source code.

    Note
    ----
    The constructor creates the range `[begin, end)`
    if the end argument is not None. Otherwise, it creates `[0, begin)`.
    """

    min: Expr
    extent: Expr
    loc: Location

    def __init__(self, begin: Expr, end: Expr | None = None, loc: Location = UnknownLoc()) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Range, begin, end, loc)

    @staticmethod
    def from_min_extent(min_value: Expr, extent: Expr, loc: Location = UnknownLoc()) -> "Range":
        """Construct a Range by min and extent.

        This constructs a range in [min_value, min_value + extent)

        Parameters
        ----------
        min_value : Expr
            The minimum value of the range.

        extent : Expr
            The extent of the range.

        loc : Location
            The location of this node in the source code.

        Returns
        -------
        rng : Range
            The constructed range.
        """
        return _ffi_api.Range_from_min_extent(min_value, extent, loc)

    def __eq__(self, other: Object) -> bool:
        return tvm_ffi.structural_equal(self, other)

    def __ne__(self, other: Object) -> bool:
        return not self.__eq__(other)


# Primitive overloads also initialize the concrete primitive nodes, whose bases
# must be defined before importing them.
from . import _overload_prim_expr  # noqa: E402  # pylint: disable=wrong-import-position

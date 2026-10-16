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
from enum import IntEnum
from functools import partial
from inspect import getattr_static
from numbers import Number

import tvm_ffi

import tvm_tirx
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from . import _ffi_api, _tensor_expr_overload
from ._constant import const
from .base import Node, Scriptable
from .type import PrimType

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
    """Base class of typed IR expressions.

    ``ty`` holds the expression's type, which may be :class:`MissingType` during
    construction. ``loc`` records its source location. Concrete subclasses define
    operands and may expose operations through their type.
    """

    loc: Location
    ty: "tvm_tirx.ir.Type"

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
        if isinstance(self.ty, tvm_tirx.ir.MissingType):
            # Preserve Relax's pre-normalization tuple access: operator calls
            # have a missing result type until the block builder infers it.
            return TupleGetItem(self, index)

        indices = tuple(index) if isinstance(index, tuple | list) else (index,)
        return _ffi_api.SubscriptExprRealize(
            self, [_convert_subscript_index(item) for item in indices], UNKNOWN_LOC
        )


@tvm_ffi.register_object("ir.StagingExpr")
class StagingExpr(Expr):
    """A traversable expression eliminated before executable IR."""


@tvm_ffi.register_object("ir.OpaqueExpr")
class OpaqueExpr(Expr):
    """Base class for opaque values that must be removed from finished IR."""


def is_prim_expr(value: object) -> bool:
    """Return whether an expression has a primitive result type."""
    return isinstance(value, Expr) and isinstance(value.ty, tvm_tirx.ir.PrimType)


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

    loc : Location, optional
        The location of this node in the source code.
    """

    name_hint: str

    def __init__(self, name_hint: str, loc: Location = UNKNOWN_LOC):
        self.__init_handle_by_constructor__(_ffi_api.GlobalVar, name_hint, loc)

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
            return _prim_expr_overload.__add__(self, other)
        result = _tensor_expr_overload.__add__(self, other)
        return result

    def __radd__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__radd__(self, other)
        result = _tensor_expr_overload.__radd__(self, other)
        return result

    def __sub__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__sub__(self, other)
        result = _tensor_expr_overload.__sub__(self, other)
        return result

    def __rsub__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rsub__(self, other)
        result = _tensor_expr_overload.__rsub__(self, other)
        return result

    def __mul__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__mul__(self, other)
        result = _tensor_expr_overload.__mul__(self, other)
        return result

    def __rmul__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rmul__(self, other)
        result = _tensor_expr_overload.__rmul__(self, other)
        return result

    def __div__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__div__(self, other)
        result = _tensor_expr_overload.__div__(self, other)
        return result

    def __rdiv__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rdiv__(self, other)
        result = _tensor_expr_overload.__rdiv__(self, other)
        return result

    def __truediv__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__truediv__(self, other)
        result = _tensor_expr_overload.__truediv__(self, other)
        return result

    def __rtruediv__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rtruediv__(self, other)
        result = _tensor_expr_overload.__rtruediv__(self, other)
        return result

    def __floordiv__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__floordiv__(self, other)
        result = _tensor_expr_overload.__floordiv__(self, other)
        return result

    def __rfloordiv__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rfloordiv__(self, other)
        result = _tensor_expr_overload.__rfloordiv__(self, other)
        return result

    def __mod__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__mod__(self, other)
        result = _tensor_expr_overload.__mod__(self, other)
        return result

    def __rmod__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rmod__(self, other)
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
            result = _prim_expr_overload.__neg__(self)
            if result is NotImplemented:
                raise TypeError("Primitive expression overload __neg__ is not registered")
            return result
        result = _tensor_expr_overload.__neg__(self)
        if result is NotImplemented:
            raise TypeError(f"Operator overloading is not supported for expression type {self.ty}")
        return result

    def __lshift__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__lshift__(self, other)
        return NotImplemented

    def __rlshift__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rlshift__(self, other)
        return NotImplemented

    def __rshift__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rshift__(self, other)
        return NotImplemented

    def __rrshift__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rrshift__(self, other)
        return NotImplemented

    def __and__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__and__(self, other)
        return NotImplemented

    def __rand__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rand__(self, other)
        return NotImplemented

    def __or__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__or__(self, other)
        return NotImplemented

    def __ror__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__ror__(self, other)
        return NotImplemented

    def __xor__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__xor__(self, other)
        return NotImplemented

    def __rxor__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__rxor__(self, other)
        return NotImplemented

    def __invert__(self):
        if is_prim_expr(self):
            result = _prim_expr_overload.__invert__(self)
            if result is NotImplemented:
                raise TypeError("Primitive expression overload __invert__ is not registered")
            return result
        raise TypeError(f"Operator overloading is not supported for expression type {self.ty}")

    def __lt__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__lt__(self, other)
        result = _tensor_expr_overload.__lt__(self, other)
        return result

    def __le__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__le__(self, other)
        result = _tensor_expr_overload.__le__(self, other)
        return result

    def __eq__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__eq__(self, other)
        return tvm_ffi.Object.__eq__(self, other)

    def __ne__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__ne__(self, other)
        return tvm_ffi.Object.__ne__(self, other)

    def __gt__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__gt__(self, other)
        result = _tensor_expr_overload.__gt__(self, other)
        return result

    def __ge__(self, other):
        if is_prim_expr(self):
            return _prim_expr_overload.__ge__(self, other)
        result = _tensor_expr_overload.__ge__(self, other)
        return result

    def __nonzero__(self):
        raise ValueError(
            "Cannot use and / or / not operator to Expr, hint: use tvm_tirx.ir.prim.all / "
            "tvm_tirx.ir.prim.any, if it is None checking, use node is not None"
        )

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other, loc: Location = UNKNOWN_LOC):
        if not is_prim_expr(self):
            raise TypeError(f"Operator overloading is not supported for expression type {self.ty}")
        result = _prim_expr_overload.equal(self, other, loc)
        if result is NotImplemented:
            raise TypeError("Primitive expression overload equal is not registered")
        return result

    def astype(self, dtype, loc: Location = UNKNOWN_LOC):
        if is_prim_expr(self):
            result = _prim_expr_overload.astype(self, dtype, loc)
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

    loc : Location, optional
        Location that points to the original source code.
    """

    fields: list[Expr]
    loc: Location

    def __init__(self, fields: list[Expr] | tuple[Expr, ...], loc: Location = UNKNOWN_LOC):
        if isinstance(fields, Tuple):
            fields = fields.fields
        elif isinstance(getattr(fields, "ty", None), tvm_tirx.ir.TupleType):
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

    loc : Location, optional
        Location that points to the original source code.
    """

    tuple_value: Expr
    index: int
    loc: Location

    def __init__(self, tuple_value: Expr, index: int, loc: Location = UNKNOWN_LOC):
        self.__init_handle_by_constructor__(_ffi_api.TupleGetItem, tuple_value, index, loc)


@tvm_ffi.register_object("ir.TensorLoad")
class TensorLoad(_CallableExprWithOp):
    """Load an element or vector from an indexed expression source.

    The source type's registered inference hook validates the inputs and derives
    the primitive result type.

    Parameters
    ----------
    source : Expr
        The tensor or other expression supporting indexed loads.
    indices : list[Expr]
        Indices interpreted by the source type, in dimension order.
    loc : Location, optional
        The source location of the load.
    """

    source: Expr
    indices: list[Expr]
    loc: Location

    def __init__(self, source: Expr, indices: list[Expr], loc: Location = UNKNOWN_LOC):
        self.__init_handle_by_constructor__(_ffi_api.TensorLoad, source, indices, loc)


@tvm_ffi.register_object("ir.Constant")
class Constant(ExprWithOp):
    """Base class of literal constants."""


@tvm_ffi.register_object("ir.FloatImm")
class FloatImm(Constant):
    """Float constant.

    Parameters
    ----------
    ty : str, tvm_ffi.dtype or PrimType
        The data type

    value : float
        The constant value.

    loc : Location, optional
        The location of this expression in the source code.
    """

    value: float

    def __init__(
        self, ty: str | tvm_ffi.dtype | PrimType, value: float, loc: Location = UNKNOWN_LOC
    ) -> None:
        if isinstance(ty, PrimType):
            ty = ty.dtype
        self.__init_handle_by_constructor__(
            _ffi_api.FloatImm,
            ty,
            value,
            loc,  # type: ignore
        )

    def __float__(self) -> float:
        return self.value


@tvm_ffi.register_object("ir.IntImm")
class IntImm(Constant):
    """Int constant.

    Parameters
    ----------
    ty : str, tvm_ffi.dtype or PrimType
        The data type

    value : int
        The constant value.

    loc : Location, optional
        The location of this expression in the source code.
    """

    value: int

    def __init__(
        self, ty: str | tvm_ffi.dtype | PrimType, value: int, loc: Location = UNKNOWN_LOC
    ) -> None:
        if isinstance(ty, PrimType):
            ty = ty.dtype
        self.__init_handle_by_constructor__(
            _ffi_api.IntImm,
            ty,
            value,
            loc,  # type: ignore
        )

    def __hash__(self) -> int:
        return self.value

    def __int__(self) -> int:
        return self.value

    def __nonzero__(self) -> bool:
        return self.value != 0

    def __eq__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpEQ(self, other, UNKNOWN_LOC)  # type: ignore

    def __ne__(self, other: Expr) -> Expr:
        return _prim_ffi_api._OpNE(self, other, UNKNOWN_LOC)  # type: ignore

    def __bool__(self) -> bool:
        return self.__nonzero__()


@tvm_ffi.register_object("ir.GenericConst")
class GenericConst(_ExprCallable, Constant):
    """A literal payload with an explicit expression type.

    Parameters
    ----------
    ty : Type
        The type assigned to the literal.
    value : object
        The payload, convertible to an FFI value.
    loc : Location, optional
        The source location of the literal.
    """

    def __init__(self, ty: "tvm_tirx.ir.Type", value, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_ffi_api.GenericConst, ty, value, loc)

    def __bool__(self) -> bool:
        return True


@tvm_ffi.register_object("ir.DataTypeImm")
class DataTypeImm(Constant):
    """A data type literal whose expression type is AnyType.

    Parameters
    ----------
    value : str or tvm_ffi.dtype
        The represented data type.
    loc : Location, optional
        The source location of the literal.
    """

    value: tvm_ffi.dtype

    def __init__(self, value: str | tvm_ffi.dtype, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DataTypeImm, value, loc)


@tvm_ffi.register_object("ir.StringImm")
class StringImm(Constant):
    """A string literal with :class:`StringType`.

    Parameters
    ----------
    value : str
        The literal text.
    loc : Location, optional
        The source location of the literal.
    """

    value: str

    def __init__(self, value: str, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_ffi_api.StringImm, value, loc)

    def __eq__(self, other) -> bool:
        return self.value == (other.value if isinstance(other, StringImm) else other)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    __hash__ = Expr.__hash__


class CallEffectKind(IntEnum):
    """Possible call effects, matching the shared C++ IR effect kinds."""

    ExprAnnotation = 0
    Pure = 1
    ReadState = 2
    UpdateState = 3
    Opaque = UpdateState
    SpecialCallArg = 4
    EmbedInfo = 5
    ControlJump = 6


@tvm_ffi.register_object("ir.Call")
class Call(_CallableExprWithOp):
    """Call a registered operator or another callable IR expression.

    Omitted or ``None`` keyword-only ``ty`` uses available result inference,
    or a missing type when no deduction is available. Explicit types, including
    ``Type.missing()``, are preserved exactly. Inference errors propagate.
    Construction permits provisional IR; :meth:`validate` checks the operator
    contract explicitly after inputs are ready. Constructing a ``Call`` never
    emits a builder statement, including calls to allocation operators.

    Parameters
    ----------
    op : Expr or str
        The callee expression, or a registered operator name resolved by ``Op.get``.
    args : sequence[Expr]
        Positional value arguments.
    attrs : Attrs or dict, optional
        Operator-specific attributes. A dictionary becomes ``DictAttrs``.
    ty_args : sequence[Type], optional
        Explicit type arguments, empty by default.
    loc : Location, optional
        The source location of the call.
    ty : Type or str, optional
        Keyword-only result type; a primitive dtype string is accepted.
    """

    op: Expr
    args: list[Expr]
    attrs: "tvm_tirx.ir.Attrs | None"
    ty_args: list["tvm_tirx.ir.Type"]
    loc: Location

    def __init__(
        self,
        op: Expr | str,
        args: list[Expr] | tuple[Expr, ...],
        attrs: "tvm_tirx.ir.Attrs | dict | None" = None,
        ty_args: list["tvm_tirx.ir.Type"] | tuple["tvm_tirx.ir.Type", ...] | None = None,
        loc: Location = UNKNOWN_LOC,
        *,
        ty: "tvm_tirx.ir.Type | str | None" = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Call, *self._normalize_constructor_args(ty, op, args, attrs, ty_args, loc)
        )

    @staticmethod
    def _normalize_constructor_args(ty, op, args, attrs, ty_args, loc):
        # pylint: disable=import-outside-toplevel
        from .attrs import DictAttrs
        from .op import Op
        from .type import PrimType, PtrType, Type

        if isinstance(op, str):
            op = Op.get(op)
        if attrs is not None and isinstance(attrs, dict):
            attrs = DictAttrs(attrs)
        if isinstance(ty, str) and ty == "handle":
            ty = PtrType(PrimType("void"))
        elif ty is not None and not isinstance(ty, Type):
            ty = PrimType(ty)
        if ty_args is None:
            ty_args = []
        return ty, op, args, attrs, ty_args, loc

    def validate(self) -> None:
        """Check the registered operator contract without changing this Call."""
        _ffi_api.CallValidate(self)


def reinfer_type(call: Call) -> "tvm_tirx.ir.Type":
    """Derive a Call's result type from its current inputs without changing the Call.

    The operator must register a fixed return type or a context-free inference
    rule. This does not invoke the operator's validator.
    """
    return _ffi_api.CallReinferType(call)


@tvm_ffi.register_object("ir.TensorRegion")
class TensorRegion(Expr, Scriptable):
    """A region of an arbitrary tensor expression.

    Parameters
    ----------
    source : Expr
        The source expression.

    region : list[Range]
        The ranges describing the region.

    ty : tvm_tirx.ir.Type
        The result type, including any dialect-specific subscript semantics.

    loc : Location, optional
        The location of the expression in the source code.
    """

    source: Expr
    region: list["Range"]

    def __init__(
        self,
        source: Expr,
        region: list["Range"],
        ty: "tvm_tirx.ir.Type",
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.TensorRegion, source, region, ty, loc)


@tvm_ffi.register_object("ir.Var")
class Var(_CallableExprWithOp):
    """A canonical local variable in the IR.

    Parameters
    ----------
    name : str, optional
        The display name. Variables with the same name remain distinct bindings.

    ty : Optional[Type or str]
        The variable type, or MissingType when omitted. A string denotes a
        primitive dtype; ``"handle"`` denotes a pointer to void.

    loc : Location, optional
        Location that points to the original source code.

    name_hint : str, optional
        Keyword-only alternative to ``name``; supply exactly one name spelling.
    """

    name: str
    loc: Location

    def __init__(
        self,
        name: str | None = None,
        ty: "tvm_tirx.ir.Type | str | None" = None,
        loc: Location = UNKNOWN_LOC,
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
        from .type import PrimType, PtrType, Type

        if isinstance(ty, str):
            ty = PtrType(PrimType("void")) if ty == "handle" else PrimType(ty)
        elif ty is not None:
            ty = tvm_ffi.convert(ty)
            if not isinstance(ty, Type):
                raise TypeError("ty must be a Type or primitive dtype string")
        self.__init_handle_by_constructor__(_ffi_api.Var, name, ty, loc)


def _lambda_type(annotation):
    """Normalize explicit types and the script scalar constructor forms."""
    if isinstance(annotation, tvm_tirx.ir.Type):
        return annotation
    if isinstance(annotation, str):
        return Var("", annotation).ty
    if isinstance(annotation, tvm_ffi.dtype):
        return tvm_tirx.ir.PrimType(annotation)
    if callable(annotation):
        value = annotation()
        if isinstance(value, Expr):
            return value.ty
        if isinstance(value, tvm_tirx.ir.Type):
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
        value = tvm_ffi.convert(value)
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
    remain in executable IR. In TVMScript, use ``T.Lambda`` to construct it.

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
    loc : Location, optional
        The location of this node in the source code.
    """

    vars: list[Var]
    body: Expr

    def __init__(
        self, parameter_types, function: Callable, *, ret_type=None, loc: Location = UNKNOWN_LOC
    ):
        variables = [Var(f"arg{i}", _lambda_type(ty)) for i, ty in enumerate(parameter_types)]
        body = _lambda_result(function(*variables))
        if ret_type is not None:
            expected = _lambda_type(ret_type)
            if not tvm_ffi.structural_equal(expected, body.ty):
                raise TypeError("LambdaExpr return annotation does not match the body type")
        self.__init_handle_by_constructor__(_ffi_api.LambdaExpr, variables, body, loc)

    def apply(self, arguments: list[Expr]) -> Expr:
        """Substitute arguments simultaneously for the lambda's bound variables."""
        return _ffi_api.LambdaExprApply(self, [_lambda_result(arg) for arg in arguments])


@tvm_ffi.register_object("ir.Range")
class Range(Node, Scriptable):
    """A half-open interval stored as ``begin`` and ``extent``.

    Use :meth:`from_min_extent` when the extent is already known.

    Parameters
    ----------
    begin : Expr
        The lower bound when ``end`` is provided; otherwise the extent of ``[0, begin)``.
    end : Expr, optional
        The exclusive upper bound. With two arguments, ``extent`` is ``end - begin``.
    loc : Location, optional
        The source location of the range.
    """

    begin: Expr
    extent: Expr
    loc: Location

    def __init__(self, begin: Expr, end: Expr | None = None, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Range, begin, end, loc)

    @staticmethod
    def from_min_extent(min_value: Expr, extent: Expr, loc: Location = UNKNOWN_LOC) -> "Range":
        """Construct a Range by min and extent.

        This constructs a range in [min_value, min_value + extent)

        Parameters
        ----------
        min_value : Expr
            The minimum value of the range.

        extent : Expr
            The extent of the range.

        loc : Location, optional
            The location of this node in the source code.

        Returns
        -------
        rng : Range
            The constructed range.
        """
        return _ffi_api.RangeFromMinExtent(min_value, extent, loc)

    @staticmethod
    def get_extent_from_begin_end(begin: Expr, end: Expr) -> Expr:
        """Extract the width from endpoints, reusing a shallow addition operand."""
        return _ffi_api.RangeGetExtentFromBeginEnd(begin, end)

    def __eq__(self, other: tvm_ffi.Object) -> bool:
        return tvm_ffi.structural_equal(self, other)

    def __ne__(self, other: tvm_ffi.Object) -> bool:
        return not self.__eq__(other)


# Primitive overloads also initialize the concrete primitive nodes, whose bases
# must be defined before importing them.
from . import _prim_expr_overload  # noqa: E402  # pylint: disable=wrong-import-position
from .prim import _ffi_api as _prim_ffi_api  # noqa: E402  # pylint: disable=wrong-import-position

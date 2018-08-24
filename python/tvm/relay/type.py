# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The type nodes of the Relay language."""
from typing import Tuple, List
from enum import IntEnum
from .base import Span, NodeBase, register_relay_node
from tvm import expr
# TODO(@jroesch): move me
from . import _make

class Type(NodeBase):
    """The base type for all Relay types."""

    def __eq__(self, other) -> bool:
        """Compares two Relay types for structural equivalence using
           alpha equivalence.
        """
        return bool(_make._type_alpha_eq(self, other))

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def same_as(self, other) -> bool:
        """Compares two Relay types by referential equality."""
        return super().__eq__(other)

@register_relay_node
class TensorType(Type):
    """A concrete TensorType in Relay, see tvm/relay/type.h for more details.
    """
    dtype: str
    shape: List[expr.Expr]
    span: Span

    def __init__(self, dtype: str, shape: List[expr.Expr]) -> None:
        self.__init_handle_by_constructor__(_make.TensorType,dtype, shape)

class Kind(IntEnum):
    """The kind of a type parameter, represents a variable shape,
       base type, type, or dimension.
    """
    ShapeVar = 0
    Shape = 1
    BaseType = 1
    Type = 2

@register_relay_node
class TypeParam(Type):
    """A type parameter used for generic types in Relay,
       see tvm/relay/type.h for more details.
    """
    var: expr.Var
    kind: Kind
    span: Span

    def __init__(self, var: expr.Var, kind: Kind) -> None:
        self.__init_handle_by_constructor__(_make.TypeParam, var, kind)

@register_relay_node
class TypeConstraint(Type):
    """Abstract class representing a type constraint."""
    pass

@register_relay_node
class FuncType(Type):
    """A function type in Relay, see tvm/relay/type.h for more details.
    """
    type_params: List[TypeParam]
    type_constraints: List[TypeConstraint]
    arg_types: List[Type]
    ret_type: Type
    span: Span

    def __init__(self, arg_types: List[Type], ret_type: Type, type_params: List[TypeParam], type_constraints: List[TypeConstraint]) -> None:
        self.__init_handle_by_constructor__(_make.FuncType, arg_types, ret_type, type_params, type_constraints)

@register_relay_node
class TypeCall(Type):
    def __init__() -> None:
        pass


@register_relay_node
class IncompleteType(Type):
    """An incomplete type."""

    def __init__(self, kind: Kind) -> None:
        self.__init_handle_by_constructor__(_make.IncompleteType, kind)

def IntType(bits: int, lanes: int=1) -> Type:
    """Constructs a integer base type.

       :param bits: The bit width of the integer type.
       :param lanes: The number of vector elements for this datatype.

    """
    return _make.IntType(bits, lanes)


def UIntType(bits: int, lanes: int=1) -> Type:
    """Constructs a unsigned integer base type.

       :param bits: The bit width of the unsigned type.
       :param lanes: The number of vector elements for this datatype.
    """
    return _make.UIntType(bits, lanes)


def FloatType(bits: int, lanes: int=1) -> Type:
    """Constructs a floating point base type.

       :param bits: The bit width of the unsigned type.
       :param lanes: The number of vector elements for this datatype.
    """
    return _make.FloatType(bits, lanes)


def BoolType(lanes: int =1) -> Type:
    """Constructs a boolean base type.

       :param bits: The bit width of the unsigned type.
       :param lanes: The number of vector elements for this datatype.
    """
    return _make.BoolType(lanes)

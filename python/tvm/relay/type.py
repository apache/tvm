# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The type nodes of the Relay language."""
from typing import Tuple, List
from enum import IntEnum
from .base import Span, NodeBase, register_relay_node
from tvm import expr

class Type(NodeBase):
    """The base type for all Relay types."""
    pass

@register_relay_node
class TensorType(Type):
    """A concrete TensorType in Relay, see tvm/relay/type.h for more details.
    """
    dtype: str
    shape: List[expr.Expr]
    span: Span

class Kind(IntEnum):
    """The kind of a type parameter, represents a variable shape,
       base type, type, or dimension.
    """
    Shape = 0
    BaseType = 1
    Type = 2
    Elem = 3

@register_relay_node
class TypeParam(Type):
    """A type parameter used for generic types in Relay,
       see tvm/relay/type.h for more details.
    """
    var: expr.Var
    kind: Kind
    span: Span

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

@register_relay_node
class IncompleteType(Type):
    """An incomplete type."""
    pass

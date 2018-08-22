# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression nodes of Relay."""
import tvm
from typing import Tuple as PyTuple, List
from enum import IntEnum
from .base import Span, NodeBase, register_relay_node
from .type import Type, TypeParam
from tvm import expr
from ._type_infer import _get_checked_type

class Expr(NodeBase):
    """The base type for all Relay exprressions."""
    def checked_type(self):
        return _get_checked_type(self)

@register_relay_node
class Constant(Expr):
    """A constant tensor in Relay, see tvm/relay/type.h for more details.
    """
    data: tvm.nd.NDArray

@register_relay_node
class Tuple(Expr):
    """A hetereogenous sequence of values.
       see tvm/relay/type.h for more details.
    """
    fields: List[Expr]

@register_relay_node
class LocalVar(Expr):
    """A local variable in Relay."""
    name_hint: str

@register_relay_node
class GlobalVar(Expr):
    """A global variable in Relay."""
    name_hint: str

@register_relay_node
class Param(Expr):
    """A function type in Relay, see tvm/relay/type.h for more details.
    """
    var: LocalVar
    type: Type

@register_relay_node
class Function(Expr):
    type_params: List[TypeParam]
    params: List[Param]
    ret_type: Type
    body: Expr

class Call(Expr):
  op: Expr
  args: List[Expr]
  # todo(@jroesch): add attrs

@register_relay_node
class Let(Expr):
    var: LocalVar
    value: Expr
    body: Expr
    value_type: Type # should be type nanotation

@register_relay_node
class If(Expr):
    cond: Expr
    true_value: Expr
    false_value: Expr
    span: Span


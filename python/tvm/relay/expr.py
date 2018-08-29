# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression nodes of Relay."""
import tvm
from typing import Tuple as PyTuple, List
from enum import IntEnum
from .base import Span, NodeBase, register_relay_node
from .type import Type, TypeParam
from tvm import expr
from ._ir_pass import _get_checked_type
from . import _make

class Expr(NodeBase):
    """The base type for all Relay exprressions."""
    def checked_type(self):
        return _get_checked_type(self)

@register_relay_node
class Constant(Expr):
    """A constant tensor in Relay, see tvm/relay/type.h for more details.
    """
    data: tvm.nd.NDArray

    def __init__(self, data: tvm.nd.NDArray) -> None:
        self.__init_handle_by_constructor__(_make.Constant, data)

@register_relay_node
class Tuple(Expr):
    """A hetereogenous sequence of values.
       see tvm/relay/type.h for more details.
    """
    fields: List[Expr]

    def __init__(self, fields: List[Expr]) -> None:
        self.__init_handle_by_constructor__(_make.Tuple, fields)


@register_relay_node
class LocalVar(Expr):
    """A local variable in Relay."""
    name_hint: str

    def __init__(self, name_hint: str) -> None:
        self.__init_handle_by_constructor__(_make.LocalVar, name_hint)

@register_relay_node
class GlobalVar(Expr):
    """A global variable in Relay."""
    name_hint: str

    def __init__(self, name_hint: str) -> None:
        self.__init_handle_by_constructor__(_make.GlobalVar, name_hint)

@register_relay_node
class Param(Expr):
    """A function type in Relay, see tvm/relay/type.h for more details.
    """
    var: LocalVar
    type: Type

    def __init__(self, var: LocalVar, type: Type) -> None:
        self.__init_handle_by_constructor__(_make.Param, var, type)


@register_relay_node
class Function(Expr):
    type_params: List[TypeParam]
    params: List[Param]
    ret_type: Type
    body: Expr

    def __init__(self, params: List[Param], ret_type: Type, body: Expr, type_params: List[TypeParam]=[]) -> None:
        self.__init_handle_by_constructor__(_make.Function, params, ret_type, body, type_params)

@register_relay_node
class Call(Expr):
  op: Expr
  args: List[Expr]
  # todo(@jroesch): add attrs

  def __init__(self, op: Expr, args: List[Expr], attrs, ty_args) -> None:
        self.__init_handle_by_constructor__(_make.Call, op, args, attrs, ty_args)

@register_relay_node
class Let(Expr):
    var: LocalVar
    value: Expr
    body: Expr
    value_type: Type # should be type nanotation

    def __init__(self, var: LocalVar, value: Expr, body: Expr, value_type: Type) -> None:
        self.__init_handle_by_constructor__(_make.Let, var, value, body, value_type)

@register_relay_node
class If(Expr):
    cond: Expr
    true_value: Expr
    false_value: Expr
    span: Span

    def __init__(self, cond: Expr, true_value: Expr, false_value: Expr) -> None:
        self.__init_handle_by_constructor__(_make.If, cond, true_value, false_value)



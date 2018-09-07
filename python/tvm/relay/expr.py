# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression nodes of Relay."""
from typing import List
import tvm
from .base import Span, NodeBase, register_relay_node
from .type import Type, TypeParam
from ._ir_pass import _get_checked_type
from . import _make


class ExprBuilder():
    """A set of methods useful for building expressions
    from other expressions.
    """
    def __call__(self, *args, **kwargs):
        converted_args = []
        for arg in args:
            import pdb
            pdb.set_trace()
            if isinstance(arg, Param):
                converted_args.append(arg.var)
            else:
                converted_args.append(arg)

        return Call(self, args, None, None)


class Expr(NodeBase, ExprBuilder):
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

    def __init__(self, var: LocalVar, ty: Type) -> None:
        self.__init_handle_by_constructor__(_make.Param, var, ty)


@register_relay_node
class Function(Expr):
    """A function in Relay, see tvm/relay/expr.h for more details."""
    type_params: List[TypeParam]
    params: List[Param]
    ret_type: Type
    body: Expr

    def __init__(self,
                 params: List[Param],
                 ret_type: Type,
                 body: Expr,
                 type_params: List[TypeParam] = None) -> None:
        if not type_params:
            type_params = []
        self.__init_handle_by_constructor__(
            _make.Function, params, ret_type, body, type_params)


@register_relay_node
class Call(Expr):
    """A function call in Relay, see tvm/relay/expr.h for more details."""
    op: Expr
    args: List[Expr]
    # todo(@jroesch): add attrs

    def __init__(self, op: Expr, args: List[Expr], attrs, ty_args=None) -> None:
        if not ty_args:
            ty_args = []

        self.__init_handle_by_constructor__(
            _make.Call, op, args, attrs, ty_args)


@register_relay_node
class Let(Expr):
    """A variable bindings in Relay, see tvm/relay/expr.h for more details."""
    var: LocalVar
    value: Expr
    body: Expr
    # should be type annotation
    value_type: Type

    def __init__(self, var: LocalVar, value: Expr, body: Expr, value_type: Type) -> None:
        self.__init_handle_by_constructor__(
            _make.Let, var, value, body, value_type)


@register_relay_node
class If(Expr):
    """A conditional expression in Relay, see tvm/relay/expr.h for more details."""
    cond: Expr
    true_value: Expr
    false_value: Expr
    span: Span

    def __init__(self, cond: Expr, true_value: Expr, false_value: Expr) -> None:
        self.__init_handle_by_constructor__(
            _make.If, cond, true_value, false_value)

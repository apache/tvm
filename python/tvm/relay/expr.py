# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression nodes of Relay."""
from typing import List
import tvm
from .base import Span, NodeBase, register_relay_node
from .ty import Type, TypeParam
from ._ir_pass import _get_checked_type
from . import _make


class Expr(NodeBase):
    """The base type for all Relay exprressions."""

    def checked_type(self):
        return _get_checked_type(self)

    def __call__(self, *args):
        converted_args = []
        for arg in args:
            if isinstance(arg, Param):
                converted_args.append(arg.var)
            else:
                converted_args.append(arg)

        return Call(self, args, None, None)


@register_relay_node
class Constant(Expr):
    """A constant tensor in Relay, see tvm/relay/type.h for more details.
    """
    data = None # type: tvm.nd.NDArray

    def __init__(self, data):
        # type: (tvm.nd.NDArray) -> None
        self.__init_handle_by_constructor__(_make.Constant, data)


@register_relay_node
class Tuple(Expr):
    """A hetereogenous sequence of values.
       see tvm/relay/type.h for more details.
    """
    fields = None # type: List[Expr]

    def __init__(self, fields):
        # type: (List[Expr]) -> None
        self.__init_handle_by_constructor__(_make.Tuple, fields)


@register_relay_node
class Var(Expr):
    """A local variable in Relay."""
    name_hint = None # type: str

    def __init__(self, name_hint):
        # type: (str) -> None
        self.__init_handle_by_constructor__(_make.Var, name_hint)


@register_relay_node
class GlobalVar(Expr):
    """A global variable in Relay."""
    name_hint = None # type: str

    def __init__(self, name_hint):
        # type: (str) -> None
        self.__init_handle_by_constructor__(_make.GlobalVar, name_hint)


@register_relay_node
class Param(Expr):
    """A function type in Relay, see tvm/relay/type.h for more details.
    """
    var = None # type: Var
    type = None # type: Type

    def __init__(self, var, ty):
        # type: (Var, Type) -> None
        self.__init_handle_by_constructor__(_make.Param, var, ty)


@register_relay_node
class Function(Expr):
    """A function in Relay, see tvm/relay/expr.h for more details."""
    type_params = None # type: List[TypeParam]
    params = None # type: List[Param]
    ret_type = None # type: Type
    body = None # type: Expr

    def __init__(self,
                 params, # type: List[Param],
                 ret_type, # type: Type,
                 body, # type: Expr,
                 type_params = None, # type: List[TypeParam]
                 ):
        # type: (...) -> None
        if not type_params:
            type_params = []
        self.__init_handle_by_constructor__(
            _make.Function, params, ret_type, body, type_params)


@register_relay_node
class Call(Expr):
    """A function call in Relay, see tvm/relay/expr.h for more details."""
    op = None # type: Expr
    args = None # type: List[Expr]
    # todo(@jroesch): add attrs

    def __init__(self, op: Expr, args: List[Expr], attrs, ty_args=None) -> None:
        if not ty_args:
            ty_args = []

        self.__init_handle_by_constructor__(
            _make.Call, op, args, attrs, ty_args)


@register_relay_node
class Let(Expr):
    """A variable bindings in Relay, see tvm/relay/expr.h for more details."""
    var: Var
    value: Expr
    body: Expr
    # should be type annotation
    value_type: Type

    def __init__(self, var: Var, value: Expr, body, value_type) -> None:
        # type: (Var, Expr, Expr, Type) -> None
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

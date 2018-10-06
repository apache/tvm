# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression nodes of Relay."""
from __future__ import absolute_import
from .base import NodeBase, register_relay_node
from . import _expr
from . import _make
from .. import convert


class Expr(NodeBase):
    """The base type for all Relay expressions."""
    @property
    def checked_type(self):
        """Get the checked type of relay.

        Returns
        -------
        checked_type : relay.Type
            The checked type.
        """
        ret = self._checked_type_
        if ret is None:
            raise ValueError("The type checker has not populated"
                             " the checked_type for this node")
        return ret

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

    def __init__(self, data):
        self.__init_handle_by_constructor__(_make.Constant, data)


@register_relay_node
class Tuple(Expr):
    """A hetereogenous sequence of values.
       see tvm/relay/type.h for more details.
    """

    def __init__(self, fields):
        self.__init_handle_by_constructor__(_make.Tuple, fields)


@register_relay_node
class Var(Expr):
    """A local variable in Relay."""

    def __init__(self, name_hint):
        self.__init_handle_by_constructor__(_make.Var, name_hint)


@register_relay_node
class GlobalVar(Expr):
    """A global variable in Relay."""

    def __init__(self, name_hint):
        self.__init_handle_by_constructor__(_make.GlobalVar, name_hint)


@register_relay_node
class Param(Expr):
    """A function type in Relay, see tvm/relay/type.h for more details.
    """

    def __init__(self, var, ty):
        self.__init_handle_by_constructor__(_make.Param, var, ty)


@register_relay_node
class Function(Expr):
    """A function in Relay, see tvm/relay/expr.h for more details."""

    def __init__(self,
                 params,
                 ret_type,
                 body,
                 type_params=None
                ):
        if type_params is None:
            type_params = convert([])

        self.__init_handle_by_constructor__(
            _make.Function, params, ret_type, body, type_params)


@register_relay_node
class Call(Expr):
    """A function call in Relay, see tvm/relay/expr.h for more details."""

    def __init__(self, op, args, attrs, ty_args=None):
        if not ty_args:
            ty_args = []

        self.__init_handle_by_constructor__(
            _make.Call, op, args, attrs, ty_args)


@register_relay_node
class Let(Expr):
    """A variable bindings in Relay, see tvm/relay/expr.h for more details."""

    def __init__(self, var, value, body, value_type):
        self.__init_handle_by_constructor__(
            _make.Let, var, value, body, value_type)


@register_relay_node
class If(Expr):
    """A conditional expression in Relay, see tvm/relay/expr.h for more details."""

    def __init__(self, cond, true_value, false_value):
        self.__init_handle_by_constructor__(
            _make.If, cond, true_value, false_value)

debug_print = _expr._debug_print

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
        """Get the checked type of tvm.relay.Expr.

        Returns
        -------
        checked_type : tvm.relay.Type
            The checked type.
        """
        ret = self._checked_type_
        if ret is None:
            raise ValueError("The type checker has not populated"
                             " the checked_type for this node")
        return ret

    def __call__(self, *args):
        return Call(self, args, None, None)


@register_relay_node
class Constant(Expr):
    """A constant expression in Relay.

    Parameters
    ----------
    data : tvm.nd.NDArray
        The data content of the constant expression.
    """
    def __init__(self, data):
        self.__init_handle_by_constructor__(_make.Constant, data)


@register_relay_node
class Tuple(Expr):
    """Tuple expression that groups several fields together.

    Parameters
    ----------
    fields : List[tvm.relay.Expr]
        The fields in the tuple.
    """
    def __init__(self, fields):
        self.__init_handle_by_constructor__(_make.Tuple, fields)


@register_relay_node
class Var(Expr):
    """A local variable in Tvm.Relay.

    Local variable can be used to declare input
    arguments to a function, or intermediate variables.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
        This name only acts as a hint, and is not used
        for equality.

    type_annotation: tvm.relay.Type, optional
        The type annotation on the variable.
    """
    def __init__(self, name_hint, type_annotation=None):
        self.__init_handle_by_constructor__(
            _make.Var, name_hint, type_annotation)


@register_relay_node
class GlobalVar(Expr):
    """A global variable in Tvm.Relay.

    GlobalVar is used to refer to the global functions
    stored in the environment.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
    """
    def __init__(self, name_hint):
        self.__init_handle_by_constructor__(_make.GlobalVar, name_hint)


@register_relay_node
class Function(Expr):
    """A function declaration expression.

    Parameters
    ----------
    params: List[tvm.relay.Var]
        List of input parameters to the function.

    ret_type: tvm.relay.Type
        The return type annotation of the function.

    body: tvm.relay.Expr
        The body of the function.

    type_params: Optional[List[tvm.relay.TypeParam]]
        The additional type parameters, this is only
        used in advanced usecase of template functions.
    """
    def __init__(self,
                 params,
                 ret_type,
                 body,
                 type_params=None):
        if type_params is None:
            type_params = convert([])

        self.__init_handle_by_constructor__(
            _make.Function, params, ret_type, body, type_params)


@register_relay_node
class Call(Expr):
    """Function call node in Relay.

    Call node corresponds the operator application node
    in computational graph terminology.

    Parameters
    ----------
    op: tvm.relay.Op or any tvm.relay.Expr with function type.
        The operation to be called.

    args: List[tvm.relay.Expr]
        The arguments to the call.

    attrs: Optional[tvm.Attrs]
        Attributes to the call, can be None

    type_args: Optional[List[tvm.relay.Type]]
        The additional type arguments, this is only
        used in advanced usecase of template functions.
    """
    def __init__(self, op, args, attrs=None, type_args=None):
        if not type_args:
            type_args = []
        self.__init_handle_by_constructor__(
            _make.Call, op, args, attrs, type_args)


@register_relay_node
class Let(Expr):
    """Let variable binding expression.

    Parameters
    ----------
    var: tvm.relay.Var
        The local variable to be bound.

    value: tvm.relay.Expr
        The value to be bound.

    body: tvm.relay.Expr
        The body of the let binding.
    """
    def __init__(self, var, value, body):
        self.__init_handle_by_constructor__(
            _make.Let, var, value, body)


@register_relay_node
class If(Expr):
    """A conditional expression in Relay.

    Parameters
    ----------
    cond: tvm.relay.Expr
        The condition.

    true_branch: tvm.relay.Expr
        The expression evaluated when condition is true.

    false_branch: tvm.relay.Expr
        The expression evaluated when condition is false.
    """
    def __init__(self, cond, true_branch, false_branch):
        self.__init_handle_by_constructor__(
            _make.If, cond, true_branch, false_branch)


@register_relay_node
class TupleGetItem(Expr):
    """Get index-th item from a tuple.

    Parameters
    ----------
    tuple_value: tvm.relay.Expr
        The input tuple expression.

    index: int
        The index.
    """
    def __init__(self, tuple_value, index):
        self.__init_handle_by_constructor__(
            _make.TupleGetItem, tuple_value, index)

debug_print = _expr._debug_print

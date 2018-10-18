# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression nodes of Relay."""
from __future__ import absolute_import

import numpy as _np
from .base import RelayNode, register_relay_node
from . import _make
from . import ty as _ty
from .._ffi import base as _base, node as _node
from .. import nd as _nd
from .. import convert


class Expr(RelayNode):
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
    """A local variable in Relay.

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

    body: tvm.relay.Expr
        The body of the function.

    ret_type: Optional[tvm.relay.Type]
        The return type annotation of the function.

    type_params: Optional[List[tvm.relay.TypeParam]]
        The additional type parameters, this is only
        used in advanced usecase of template functions.
    """
    def __init__(self,
                 params,
                 body,
                 ret_type=None,
                 type_params=None):
        if type_params is None:
            type_params = convert([])

        self.__init_handle_by_constructor__(
            _make.Function, params, body, ret_type, type_params)


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
    variable: tvm.relay.Var
        The local variable to be bound.

    value: tvm.relay.Expr
        The value to be bound.

    body: tvm.relay.Expr
        The body of the let binding.
    """
    def __init__(self, variable, value, body):
        self.__init_handle_by_constructor__(
            _make.Let, variable, value, body)


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


class TupleWrapper(_node.NodeGeneric):
    """TupleWrapper.

    This class is a Python wrapper for a Relay tuple of known size.
    It allows for accessing the fields of the Relay tuple as though
    it were a Python tuple.

    Parameters
    ----------
    tuple_value: tvm.relay.Expr
        The input tuple

    size: int
        The size of the tuple.
    """
    def __init__(self, tuple_value, size):
        self.tuple_value = tuple_value
        self.size = size

    def asnode(self):
        """Returns the underlying Relay tuple if this wrapper is passed
        as an argument to an FFI function."""

        return self.tuple_value

    def __getitem__(self, key):
        return self.tuple_value.fields[key]

    def __len__(self):
        return len(self.tuple_value.fields)


def var(name_hint,
        type_annotation=None,
        shape=None,
        dtype="float32"):
    """Create a new tvm.relay.Var.

    This is a simple wrapper function that allows specify
    shape and dtype directly.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
        This name only acts as a hint, and is not used
        for equality.

    type_annotation: Optional[tvm.relay.Type, str]
        The type annotation on the variable.
        When type_annotation is a str, we will create a scalar variable.

    shape: Optional[List[tvm.Expr]]
        The shape of the tensor type.

    dtype: str, optional
        The data type of the tensor.

    Examples
    --------
    .. code-block:: python

      # The following 4 lines are equivalent to each other
      x = tvm.relay.Var("x", tvm.relay.TensorType([1, 2]))
      x = tvm.relay.var("x", tvm.relay.TensorType([1, 2]))
      x = tvm.relay.var("x", shape=[1, 2])
      x = tvm.relay.var("x", shape=[1, 2], dtype="float32")

      # The following 2 lines are equivalent to each other.
      y = tvm.relay.var("x", "float32")
      y = tvm.relay.var("x", shape=(), dtype="float32")
    """
    if type_annotation is not None and shape is not None:
        raise ValueError("Can only specify either type_annotation or shape.")
    if shape is not None:
        type_annotation = _ty.TensorType(shape, dtype)
    elif isinstance(type_annotation, str):
        type_annotation = _ty.TensorType((), type_annotation)
    return Var(name_hint, type_annotation)


def const(value, dtype=None):
    """Create a constant value.

    Parameters
    ----------
    value: Union[bool, int, float, numpy.ndarray, tvm.nd.NDArray]
        The constant value.

    dtype: str, optional
        The data type of the value.
    """
    if isinstance(value, _base.numeric_types):
        value = _np.array(value, dtype=dtype)
    elif isinstance(value, (bool, list)):
        value = _np.array(value, dtype=dtype)
    if isinstance(value, (_np.ndarray, _np.generic)):
        value = _nd.array(value)
    if not isinstance(value, _nd.NDArray):
        raise ValueError("value has to be scalar or NDArray")
    return Constant(value)

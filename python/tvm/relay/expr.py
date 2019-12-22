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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression nodes of Relay."""
from __future__ import absolute_import
from numbers import Number as _Number

import numpy as _np
from .base import RelayNode, register_relay_node
from . import _make
from . import _expr
from . import ty as _ty
from .._ffi import base as _base
from .. import nd as _nd
from .. import convert
from ..ndarray import NDArray

# will be registered afterwards
_op_make = None

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

    def astype(self, dtype):
        """Cast the content type of the current data to dtype.

        Parameters
        ----------
        dtype : str
            The target data type.

        Note
        ----
        This function only works for TensorType Exprs.

        Returns
        -------
        result : tvm.relay.Expr
            The result expression.
        """
        return _make.cast(self, dtype)

    def __neg__(self):
        return _op_make.negative(self)

    def __lt__(self, other):
        if isinstance(other, Expr):
            return _op_make.less(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __gt__(self, other):
        if isinstance(other, Expr):
            return _op_make.greater(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __ge__(self, other):
        if isinstance(other, Expr):
            return _op_make.greater_equal(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __le__(self, other):
        if isinstance(other, Expr):
            return _op_make.less_equal(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __add__(self, other):
        if isinstance(other, Expr):
            return _op_make.add(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Expr):
            return _op_make.subtract(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __rsub__(self, other):
        if isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __mul__(self, other):
        if isinstance(other, Expr):
            return _op_make.multiply(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, Expr):
            return _op_make.divide(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __rdiv__(self, other):
        if isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __call__(self, *args):
        """Call the variable (if it represents a function).

        Parameters
        ----------
        args: List[relay.Expr]
            The arguments to the call.

        Returns
        -------
        call: Call
            A call taking the variable as a function.
        """
        return Call(self, args)

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

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Tuple index out of range")
        return self.fields[index]

    def __len__(self):
        return len(self.fields)

    def astype(self, _):
        raise TypeError("astype cannot be used on tuple")


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

    @property
    def name_hint(self):
        """Get name hint of the current var."""
        name = self.vid.name_hint
        return name


@register_relay_node
class GlobalVar(Expr):
    """A global variable in Tvm.Relay.

    GlobalVar is used to refer to the global functions
    stored in the module.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
    """
    def __init__(self, name_hint):
        self.__init_handle_by_constructor__(_make.GlobalVar, name_hint)

    def __call__(self, *args):
        """Invoke the gobal function.

        Parameters
        ----------
        args: List[relay.Expr]
            Arguments.
        """
        return Call(self, args, None, None)


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
                 type_params=None,
                 attrs=None):
        if type_params is None:
            type_params = convert([])

        self.__init_handle_by_constructor__(
            _make.Function, params, body, ret_type, type_params, attrs)

    def __call__(self, *args):
        """Invoke the global function.

        Parameters
        ----------
        args: List[relay.Expr]
            Arguments.
        """
        return Call(self, args, None, None)

    def get_params(self):
        return _expr.FunctionGetParams(self)

    def set_params(self, params):
        for key in params:
            value = params[key]
            if isinstance(value, NDArray):
                params[key] = Constant(value)

        return _expr.FunctionSetParams(self, params)

    def set_attribute(self, name, ref):
        return _expr.FunctionSetAttr(self, name, ref)


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


@register_relay_node
class RefCreate(Expr):
    """Create a new reference from initial value.
    Parameters
    ----------
    value: tvm.relay.Expr
       The initial value.
    """
    def __init__(self, value):
        self.__init_handle_by_constructor__(_make.RefCreate, value)


@register_relay_node
class RefRead(Expr):
    """Get the value inside the reference.
    Parameters
    ----------
    ref: tvm.relay.Expr
         The reference.
    """
    def __init__(self, ref):
        self.__init_handle_by_constructor__(_make.RefRead, ref)


@register_relay_node
class RefWrite(Expr):
    """
    Update the value inside the reference.
    The whole expression will evaluate to an empty tuple.
    Parameters
    ----------
    ref: tvm.relay.Expr
        The reference.
    value: tvm.relay.Expr
        The new value.
    """
    def __init__(self, ref, value):
        self.__init_handle_by_constructor__(_make.RefWrite, ref, value)


class TempExpr(Expr):
    """Baseclass of all TempExpr.

    TempExprs are pass specific expression that can be
    useful to define intermediate result in the
    rewriting pass such as layout or type transformation.
    """
    def realize(self):
        """Convert the expression to a normal(non-temp) Expr.

        Returns
        -------
        The corresponding normal expression.
        """
        return _expr.TempExprRealize(self)


class TupleWrapper(object):
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

    def astuple(self):
        """Returns the underlying Relay tuple if this wrapper is passed
        as an argument to an FFI function."""
        return self.tuple_value

    def astext(self):
        """Get the text format of the tuple expression.

        Returns
        -------
        text : str
            The text format of the tuple expression.
        """
        return self.tuple_value.astext()

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Tuple index out of range")
        return TupleGetItem(self.tuple_value, index)

    def __len__(self):
        return self.size

    def __repr__(self):
        return ("TupleWrapper(" + self.tuple_value.__repr__() +
                ", " + str(self.size) + ")")

    def astype(self, _):
        raise TypeError("astype cannot be used on tuple")


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

    Note
    ----
    When dtype is None, we use the following rule:

    - int maps to "int32"
    - float maps to "float32"
    - bool maps to "bool"
    - other using the same default rule as numpy.
    """
    if isinstance(value, (_base.numeric_types, (bool, list))):
        value = _np.array(value, dtype=dtype)

    if not dtype:
        # when dtype is None: int maps to "int32", float maps to "float32"
        map_dtype = {
            _np.dtype('int64'): _np.int32,
            _np.dtype('float64'): _np.float32
            }.get(value.dtype, None)
        if map_dtype:
            value = value.astype(map_dtype)

    if isinstance(value, (_np.ndarray, _np.generic)):
        value = _nd.array(value)

    if not isinstance(value, _nd.NDArray):
        raise ValueError("value has to be scalar or NDArray")
    return Constant(value)


def bind(expr, binds):
    """Bind an free variables in expr or function arguments.

    We can bind parameters expr if it is a function.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    binds : Union[Map[tvm.relay.Var, tvm.relay.Expr], Map[str, tvm.relay.Expr]]
        The specific bindings.

    Returns
    -------
    result : tvm.relay.Expr
        The expression or function after binding.
    """
    return _expr.Bind(expr, binds)

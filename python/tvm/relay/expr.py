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
# pylint: disable=no-else-return, invalid-name, unused-import
"""The expression nodes of Relay."""
from __future__ import absolute_import
from numbers import Number as _Number

import numpy as _np
import tvm._ffi
from tvm._ffi import base as _base
from tvm.runtime import NDArray, ndarray as _nd
from tvm.ir import RelayExpr, GlobalVar, Node

from .base import RelayNode
from . import _ffi_api
from . import ty as _ty

# alias relay expr as Expr.
Expr = RelayExpr

# will be registered afterwards
_op_make = None


class ExprWithOp(RelayExpr):
    """Basetype of all relay expressions that defines op overloading."""

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
        return _ffi_api.cast(self, dtype)

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


@tvm._ffi.register_object("relay.Constant")
class Constant(ExprWithOp):
    """A constant expression in Relay.

    Parameters
    ----------
    data : tvm.nd.NDArray
        The data content of the constant expression.
    """

    def __init__(self, data):
        self.__init_handle_by_constructor__(_ffi_api.Constant, data)


@tvm._ffi.register_object("relay.Tuple")
class Tuple(ExprWithOp):
    """Tuple expression that groups several fields together.

    Parameters
    ----------
    fields : List[tvm.relay.Expr]
        The fields in the tuple.

    span: Optional[tvm.relay.Span]
        Span that points to original source code
    """

    def __init__(self, fields, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Tuple, fields, span)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Tuple index out of range")
        return self.fields[index]

    def __len__(self):
        return len(self.fields)

    def astype(self, _):
        raise TypeError("astype cannot be used on tuple")


@tvm._ffi.register_object("relay.Var")
class Var(ExprWithOp):
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
        self.__init_handle_by_constructor__(_ffi_api.Var, name_hint, type_annotation)

    @property
    def name_hint(self):
        """Get name hint of the current var."""
        name = str(self.vid.name_hint)
        return name


@tvm._ffi.register_object("relay.Call")
class Call(ExprWithOp):
    """Function call node in Relay.

    Call node corresponds the operator application node
    in computational graph terminology.

    Parameters
    ----------
    op: tvm.ir.Op or any tvm.relay.Expr with function type.
        The operation to be called.

    args: List[tvm.relay.Expr]
        The arguments to the call.

    attrs: Optional[tvm.Attrs]
        Attributes to the call, can be None

    type_args: Optional[List[tvm.relay.Type]]
        The additional type arguments, this is only
        used in advanced usecase of template functions.

    span: Optional[tvm.relay.Span]
        Span that points to original source code
    """

    def __init__(self, op, args, attrs=None, type_args=None, span=None):
        if not type_args:
            type_args = []
        self.__init_handle_by_constructor__(_ffi_api.Call, op, args, attrs, type_args, span)


@tvm._ffi.register_object("relay.Let")
class Let(ExprWithOp):
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
        self.__init_handle_by_constructor__(_ffi_api.Let, variable, value, body)


@tvm._ffi.register_object("relay.If")
class If(ExprWithOp):
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
        self.__init_handle_by_constructor__(_ffi_api.If, cond, true_branch, false_branch)


@tvm._ffi.register_object("relay.TupleGetItem")
class TupleGetItem(ExprWithOp):
    """Get index-th item from a tuple.

    Parameters
    ----------
    tuple_value: tvm.relay.Expr
        The input tuple expression.

    index: int
        The index.
    """

    def __init__(self, tuple_value, index):
        self.__init_handle_by_constructor__(_ffi_api.TupleGetItem, tuple_value, index)


@tvm._ffi.register_object("relay.RefCreate")
class RefCreate(ExprWithOp):
    """Create a new reference from initial value.
    Parameters
    ----------
    value: tvm.relay.Expr
       The initial value.
    """

    def __init__(self, value):
        self.__init_handle_by_constructor__(_ffi_api.RefCreate, value)


@tvm._ffi.register_object("relay.RefRead")
class RefRead(ExprWithOp):
    """Get the value inside the reference.
    Parameters
    ----------
    ref: tvm.relay.Expr
         The reference.
    """

    def __init__(self, ref):
        self.__init_handle_by_constructor__(_ffi_api.RefRead, ref)


@tvm._ffi.register_object("relay.RefWrite")
class RefWrite(ExprWithOp):
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
        self.__init_handle_by_constructor__(_ffi_api.RefWrite, ref, value)


class TempExpr(ExprWithOp):
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
        return _ffi_api.TempExprRealize(self)


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
        return "TupleWrapper(" + self.tuple_value.__repr__() + ", " + str(self.size) + ")"

    def astype(self, _):
        raise TypeError("astype cannot be used on tuple")


def var(name_hint, type_annotation=None, shape=None, dtype="float32"):
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
        The data type of the resulting constant.

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
        dtype = {_np.dtype("int64"): _np.int32, _np.dtype("float64"): _np.float32}.get(
            value.dtype, None
        )

    if isinstance(value, (_np.ndarray, _np.generic)):
        if dtype is not None:
            value = value.astype(dtype)
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

    binds : Map[tvm.relay.Var, tvm.relay.Expr]
        The specific bindings.

    Returns
    -------
    result : tvm.relay.Expr
        The expression or function after binding.
    """
    return _ffi_api.Bind(expr, binds)


@tvm._ffi.register_object("relay.StorageInfo")
class StorageInfo(Node):
    """StorageInfo

    The static storage information produced by memory planning.
    Contains the storage ids where expressions are stored, the
    type of the "virtual devices" the expressions are stored on,
    and the sizes of each storage element."""

    def __init__(self, sids, dev_types, sizes):
        self.__init_handle_by_constructor__(_ffi_api.StorageInfo, sids, dev_types, sizes)

    @property
    def storage_ids(self):
        return _ffi_api.StorageInfoStorageIds(self)

    @property
    def device_types(self):
        return _ffi_api.StorageInfoDeviceTypes(self)

    @property
    def storage_sizes(self):
        return _ffi_api.StorageInfoStorageSizes(self)

    @property
    def virtual_devices(self):
        return _ffi_api.StorageInfoVirtualDevices(self)


@tvm._ffi.register_object("relay.StaticMemoryPlan")
class StaticMemoryPlan(Node):
    """StaticMemoryPlan

    The result of static memory planning."""

    def __init__(self, expr_to_storage_info):
        self.__init_handle_by_constructor__(_ffi_api.StaticMemoryPlan, expr_to_storage_info)

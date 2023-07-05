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
from tvm.ir import GlobalVar, Node, RelayExpr
from tvm.runtime import NDArray
from tvm.runtime import ndarray as _nd

from . import _ffi_api
from . import ty as _ty
from .base import RelayNode, astext, pretty_print

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

    def __str__(self):
        return pretty_print(self)

    def astext(self, show_meta_data=True, annotate=None):
        """Get the text format of the expression.

        Parameters
        ----------
        show_meta_data : bool
            Whether to include meta data section in the text
            if there is meta data.

        annotate: Optional[Object->str]
            Optionally annotate function to provide additional
            information in the comment block.

        Returns
        -------
        text : str
            The text format of the expression.

        Notes
        -----
        The meta data section is necessary to fully parse the text format.
        However, it can contain dumps that are big (e.g constant weights),
        so it can be helpful to skip printing the meta data section.
        """
        return astext(self, show_meta_data, annotate)

    def __neg__(self):
        return _op_make.negative(self)

    def __lt__(self, other):
        if isinstance(other, Expr):
            return _op_make.less(self, other)
        elif isinstance(other, _Number):
            raise TypeError(f'convert "{str(other)}" with `const` first')
        else:
            raise TypeError(f"type {type(other)} not supported")

    def __gt__(self, other):
        if isinstance(other, Expr):
            return _op_make.greater(self, other)
        elif isinstance(other, _Number):
            raise TypeError(f'convert "{str(other)}" with `const` first')
        else:
            raise TypeError(f"type {type(other)} not supported")

    def __ge__(self, other):
        if isinstance(other, Expr):
            return _op_make.greater_equal(self, other)
        elif isinstance(other, _Number):
            raise TypeError(f'convert "{str(other)}" with `const` first')
        else:
            raise TypeError(f"type {type(other)} not supported")

    def __le__(self, other):
        if isinstance(other, Expr):
            return _op_make.less_equal(self, other)
        elif isinstance(other, _Number):
            raise TypeError(f'convert "{str(other)}" with `const` first')
        else:
            raise TypeError(f"type {type(other)} not supported")

    def __add__(self, other):
        if isinstance(other, Expr):
            return _op_make.add(self, other)
        elif isinstance(other, _Number):
            raise TypeError(f'convert "{str(other)}" with `const` first')
        else:
            raise TypeError(f"type {type(other)} not supported")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Expr):
            return _op_make.subtract(self, other)
        elif isinstance(other, _Number):
            raise TypeError(f'convert "{str(other)}" with `const` first')
        else:
            raise TypeError(f"type {type(other)} not supported")

    def __rsub__(self, other):
        if isinstance(other, _Number):
            raise TypeError(f'convert "{str(other)}" with `const` first')
        raise TypeError(f"type {type(other)} not supported")

    def __mul__(self, other):
        if isinstance(other, Expr):
            return _op_make.multiply(self, other)
        elif isinstance(other, _Number):
            raise TypeError(f'convert "{str(other)}" with `const` first')
        else:
            raise TypeError(f"type {type(other)} not supported")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, Expr):
            return _op_make.divide(self, other)
        elif isinstance(other, _Number):
            raise TypeError(f'convert "{str(other)}" with `const` first')
        else:
            raise TypeError(f"type {type(other)} not supported")

    def __rdiv__(self, other):
        if isinstance(other, _Number):
            raise TypeError(f'convert "{str(other)}" with `const` first')
        raise TypeError(f"type {type(other)} not supported")

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

    span: Optional[tvm.relay.Span]
        Span that points to original source code.
    """

    def __init__(self, data, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Constant, data, span)


@tvm._ffi.register_func("relay.ConstantWithFields")
def ConstantWithFields(constant, data=None, virtual_device=None, span=None):
    """
    Returns constant with the given properties. A None property denotes 'no change'.
    Returns constant if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.ConstantWithFields(constant, data, virtual_device, span)


@tvm._ffi.register_object("relay.Tuple")
class Tuple(ExprWithOp):
    """Tuple expression that groups several fields together.

    Parameters
    ----------
    fields : List[tvm.relay.Expr]
        The fields in the tuple.

    span: Optional[tvm.relay.Span]
        Span that points to original source code.
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


@tvm._ffi.register_func("relay.TupleWithFields")
def TupleWithFields(tup, fields=None, virtual_device=None, span=None):
    """
    Returns tuple with the given properties. A None property denotes 'no change'.
    Returns tuple if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.TupleWithFields(tup, fields, virtual_device, span)


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

    span: Optional[tvm.relay.Span]
        Span that points to original source code.
    """

    def __init__(self, name_hint, type_annotation=None, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Var, name_hint, type_annotation, span)

    @property
    def name_hint(self):
        """Get name hint of the current var."""
        name = str(self.vid.name_hint)
        return name


@tvm._ffi.register_func("relay.VarWithFields")
def VarWithFields(variable, vid=None, type_annotation=None, virtual_device=None, span=None):
    """
    Returns var with the given properties. A None property denotes 'no change'.
    Returns var if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.VarWithFields(variable, vid, type_annotation, virtual_device, span)


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
        Span that points to original source code.
    """

    def __init__(self, op, args, attrs=None, type_args=None, span=None):
        if not type_args:
            type_args = []
        self.__init_handle_by_constructor__(_ffi_api.Call, op, args, attrs, type_args, span)


@tvm._ffi.register_func("relay.CallWithFields")
def CallWithFields(
    call, op=None, args=None, attrs=None, type_args=None, virtual_device=None, span=None
):
    """
    Returns call with the given properties. A None property denotes 'no change'.
    Returns call if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.CallWithFields(call, op, args, attrs, type_args, virtual_device, span)


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

    span: Optional[tvm.relay.Span]
        Span that points to original source code.
    """

    def __init__(self, variable, value, body, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Let, variable, value, body, span)


@tvm._ffi.register_func("relay.LetWithFields")
def LetWithFields(let, variable=None, value=None, body=None, virtual_device=None, span=None):
    """
    Returns let with the given properties. A None property denotes 'no change'.
    Returns let if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.LetWithFields(let, variable, value, body, virtual_device, span)


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

    span: Optional[tvm.relay.Span]
        Span that points to original source code.
    """

    def __init__(self, cond, true_branch, false_branch, span=None):
        self.__init_handle_by_constructor__(_ffi_api.If, cond, true_branch, false_branch, span)


@tvm._ffi.register_func("relay.IfWithFields")
def IfWithFields(
    if_expr, cond=None, true_branch=None, false_branch=None, virtual_device=None, span=None
):
    """
    Returns if with the given properties. A None property denotes 'no change'.
    Returns if if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.IfWithFields(if_expr, cond, true_branch, false_branch, virtual_device, span)


@tvm._ffi.register_object("relay.TupleGetItem")
class TupleGetItem(ExprWithOp):
    """Get index-th item from a tuple.

    Parameters
    ----------
    tuple_value: tvm.relay.Expr
        The input tuple expression.

    index: int
        The index.

    span: Optional[tvm.relay.Span]
        Span that points to original source code.
    """

    def __init__(self, tuple_value, index, span=None):
        self.__init_handle_by_constructor__(_ffi_api.TupleGetItem, tuple_value, index, span)


@tvm._ffi.register_func("relay.TupleGetItemWithFields")
def TupleGetItemWithFields(
    tuple_get_item, tuple_value=None, index=None, virtual_device=None, span=None
):
    """
    Returns tuple_get_item with the given properties. A None property denotes 'no change'.
    Returns tuple_get_item if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.TupleGetItemWithFields(tuple_get_item, tuple_value, index, virtual_device, span)


@tvm._ffi.register_object("relay.RefCreate")
class RefCreate(ExprWithOp):
    """Create a new reference from initial value.
    Parameters
    ----------
    value: tvm.relay.Expr
       The initial value.

    span: Optional[tvm.relay.Span]
        Span that points to original source code.
    """

    def __init__(self, value, span=None):
        self.__init_handle_by_constructor__(_ffi_api.RefCreate, value, span)


@tvm._ffi.register_func("relay.RefCreateWithFields")
def RefCreateWithFields(ref_create, value=None, virtual_device=None, span=None):
    """
    Returns ref_create with the given properties. A None property denotes 'no change'.
    Returns ref_create if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.RefCreateWithFields(ref_create, value, virtual_device, span)


@tvm._ffi.register_object("relay.RefRead")
class RefRead(ExprWithOp):
    """Get the value inside the reference.
    Parameters
    ----------
    ref: tvm.relay.Expr
         The reference.

    span: Optional[tvm.relay.Span]
        Span that points to original source code.
    """

    def __init__(self, ref, span=None):
        self.__init_handle_by_constructor__(_ffi_api.RefRead, ref, span)


@tvm._ffi.register_func("relay.RefReadWithFields")
def RefReadWithFields(ref_read, ref=None, virtual_device=None, span=None):
    """
    Returns ref_read with the given properties. A None property denotes 'no change'.
    Returns ref_read if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.RefReadWithFields(ref_read, ref, virtual_device, span)


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

    span: Optional[tvm.relay.Span]
        Span that points to original source code.
    """

    def __init__(self, ref, value, span=None):
        self.__init_handle_by_constructor__(_ffi_api.RefWrite, ref, value, span)


@tvm._ffi.register_func("relay.RefWriteWithFields")
def RefWriteWithFields(ref_write, ref=None, value=None, virtual_device=None, span=None):
    """
    Returns ref_write with the given properties. A None property denotes 'no change'.
    Returns ref_write if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.RefWriteWithFields(ref_write, ref, value, virtual_device, span)


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
        return TupleGetItem(self.tuple_value, index, span=self.tuple_value.span)

    def __len__(self):
        return self.size

    def __repr__(self):
        return "TupleWrapper(" + self.tuple_value.__repr__() + ", " + str(self.size) + ")"

    def astype(self, _):
        raise TypeError("astype cannot be used on tuple")


def var(name_hint, type_annotation=None, shape=None, dtype="float32", span=None):
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

    span: Optional[tvm.relay.Span]
        Span that points to original source code.

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
    return Var(name_hint, type_annotation, span)


def const(value, dtype=None, span=None):
    """Create a constant value.

    Parameters
    ----------
    value: Union[bool, int, float, numpy.ndarray, tvm.nd.NDArray]
        The constant value.

    dtype: str, optional
        The data type of the resulting constant.

    span: Optional[tvm.relay.Span]
        Span that points to original source code.

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

    return Constant(value, span)


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

    def __str__(self):
        return pretty_print(self)

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

    def __str__(self):
        return pretty_print(self)

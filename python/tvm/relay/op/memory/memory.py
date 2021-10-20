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
# pylint: disable=no-else-return,invalid-name,len-as-condition,too-many-nested-blocks
"""Operators for manipulating low-level memory."""
from __future__ import absolute_import as _abs
from . import _make


def alloc_tensor(storage, offset, shape, dtype="float32", assert_shape=None):
    """Allocate a tensor with the provided shape, and dtype.

    Parameters
    ----------
    storage : tvm.relay.Expr
        The storage to allocate from.

    offset : tvm.relay.Expr
        The offset to allocate from.

    shape : tvm.relay.Expr
        The shape of the tensor to allocate.

    dtype: str
        The dtype of the tensor.

    assert_shape: Control the static shape when computed by dynamic shape expression.

    Returns
    -------
    result : tvm.relay.Expr
        The alloc_tensor expression.
    """
    return _make.alloc_tensor(storage, offset, shape, dtype, assert_shape)


def alloc_storage(size, alignment, device, dtype_hint="float32"):
    """Allocate a piece of tensor storage.

    Parameters
    ----------
    size : tvm.relay.Expr
        The size of the allocation.
    alignment : tvm.relay.Expr
        The alignment of the allocation.
    device : tvm.runtime.Device
        The device of the allocation.
    dtype_hint : str
        The dtype hint of the allocation.

    Returns
    -------
    result : tvm.relay.Expr
        The alloc_storage expression.
    """
    return _make.alloc_storage(size, alignment, device, dtype_hint)


def flatten_tuple_type(ty):
    """Return a sequence of the types contained in the tuple type in order.

    Parameters
    ----------
    ty: tvm.Type
        The type to flatten.

    Returns
    -------
    result: List[tvm.Type]
        The types in their linear order.
    """
    return _make.FlattenTupleType(ty)


def from_tuple_type(ty, expr):
    """Convert an expression with the given type into a sequence of expressions.
       Each expression maps to a field of the tuple or nested tuples in linear
       order.

    Parameters
    ----------
    ty: tvm.Type
        The type to unpack.

    expr: tvm.relay.Expr
        The expression from which to extract each sub-field.

    Returns
    -------
    result: List[tvm.relay.Expr]
        The list of sub-expressions.
    """
    return _make.FromTupleType(ty, expr)


def to_tuple_type(ty, exprs):
    """Pack the sequence of expressions into the nested tuple type.

    Parameters
    ----------
    ty: tvm.Type
        The type to pack with.

    exprs: tvm.relay.Expr
        The expressions to pack back into the nested tuple type.

    Returns
    -------
    result: List[tvm.relay.Expr]
        The packed tuple expression.
    """
    return _make.ToTupleType(ty, exprs)

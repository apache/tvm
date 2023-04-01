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
# pylint: disable=wildcard-import, redefined-builtin
"""Relax builtin operators."""
from tvm import _ffi

from ..expr import Call
from . import ty
from . import ty_guard as tg


## (TVM-TOOL) py_op begin memory/*
def alloc_storage(
    size: ty.Shape,
    virtual_device_index: ty.IntPrimExpr,
    storage_scope: ty.Str,
    dtype: ty.DType,
) -> Call:
    """Allocate a chunk of memory storage with specific size, dtype on a specific device
        on its specific storage scope. The allocated storage can be used to create tensors in-place.
        The storage will only be freed when the program exits or when the storage is killed by
        R.memory.kill_storage.

    Parameters
    ----------
    size : ty.Shape
        The shape of the storage.
    virtual_device_index : ty.IntPrimExpr
        The index of the device on which the storage is allocated.
    storage_scope : ty.Str
        The storage scope of the storage.
    dtype : ty.DType
        The data type of the storage.

    Returns
    -------
    ret : ty.AnyRelaxExpr
        The allocated storage.
    """
    size = tg.check(0, "size", tg.Shape(), size)
    virtual_device_index = tg.check(
        1, "virtual_device_index", tg.IntPrimExpr(), virtual_device_index
    )
    storage_scope = tg.check(2, "storage_scope", tg.Str(), storage_scope)
    dtype = tg.check(3, "dtype", tg.DType(), dtype)
    _ffi_func = _ffi.get_global_func("relax.op.memory.alloc_storage")
    return _ffi_func(size, virtual_device_index, storage_scope, dtype)


def alloc_tensor(
    storage: ty.AnyRelaxExpr,
    offset: ty.IntPrimExpr,
    shape: ty.Shape,
    dtype: ty.DType,
) -> Call:
    """Allocate a tensor with specific shape, dtype on a specific device at the specific offset
        on a storage created by R.memory.alloc_storage.
        The tensor will only be freed when the program exits or when the tensor is killed by
        R.memory.kill_tensor.

    Parameters
    ----------
    storage : ty.AnyRelaxExpr
        The storage on which the tensor is allocated.
    offset : ty.IntPrimExpr
        The offset of the tensor on the storage.
    shape : ty.Shape
        The shape of the tensor.
    dtype : ty.DType
        The data type of the tensor.

    Returns
    -------
    ret : ty.Tensor
        The allocated tensor.
    """
    storage = tg.check(0, "storage", tg.AnyRelaxExpr(), storage)
    offset = tg.check(1, "offset", tg.IntPrimExpr(), offset)
    shape = tg.check(2, "shape", tg.Shape(), shape)
    dtype = tg.check(3, "dtype", tg.DType(), dtype)
    _ffi_func = _ffi.get_global_func("relax.op.memory.alloc_tensor")
    return _ffi_func(storage, offset, shape, dtype)


def kill_storage(
    storage: ty.AnyRelaxExpr,
) -> Call:
    """Kill a storage created by R.memory.alloc_storage.

    Parameters
    ----------
    storage : ty.AnyRelaxExpr
        The storage being allocated by R.memory.alloc_storage.

    Returns
    -------
    ret : ty.AnyRelaxExpr
        The call node created.
    """
    storage = tg.check(0, "storage", tg.AnyRelaxExpr(), storage)
    _ffi_func = _ffi.get_global_func("relax.op.memory.kill_storage")
    return _ffi_func(storage)


def kill_tensor(
    tensor: ty.Tensor,
) -> Call:
    """Kill a tensor created by R.memory.alloc_tensor.

    Parameters
    ----------
    tensor : ty.Tensor
        The tensor being allocated by R.memory.alloc_tensor.

    Returns
    -------
    ret : ty.AnyRelaxExpr
        The call node created.
    """
    tensor = tg.check(0, "tensor", tg.Tensor([]), tensor)
    _ffi_func = _ffi.get_global_func("relax.op.memory.kill_tensor")
    return _ffi_func(tensor)


## (TVM-TOOL) py_op end memory/*

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


## (TVM-TOOL) py_op begin vm/*
def alloc_storage(
    size: ty.Shape,
    runtime_device_index: ty.IntPrimExpr,
    dtype: ty.DType,
) -> Call:
    """Allocate a storage with specific size and dtype on a specific device.
        The allocated storage can be used to create tensors in-place.
        The storage is automatically managed by the VM.

    Parameters
    ----------
    size : ty.Shape
        The shape of the storage.
    runtime_device_index : ty.IntPrimExpr
        The index of the device on which the storage is allocated.
    dtype : ty.DType
        The data type of the storage.

    Returns
    -------
    ret : ty.AnyRelaxExpr
        The allocated storage.
    """
    size = tg.check(0, "size", tg.Shape(), size)
    runtime_device_index = tg.check(
        1, "runtime_device_index", tg.IntPrimExpr(), runtime_device_index
    )
    dtype = tg.check(2, "dtype", tg.DType(), dtype)
    _ffi_func = _ffi.get_global_func("relax.op.vm.alloc_storage")
    return _ffi_func(size, runtime_device_index, dtype)


def alloc_tensor(
    storage: ty.AnyRelaxExpr,
    offset: ty.IntPrimExpr,
    shape: ty.Shape,
    dtype: ty.DType,
) -> Call:
    """Allocate a tensor with specific shape, dtype on a specific device at the specific offset
        on a storage created by R.vm.alloc_storage. The tensor is automatically managed by the VM.

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
    _ffi_func = _ffi.get_global_func("relax.op.vm.alloc_tensor")
    return _ffi_func(storage, offset, shape, dtype)


def call_tir_dyn(
    func: ty.ExternFunc,
    args: ty.TupleExpr,
) -> Call:
    """Call a TIR function with dynamic arguments.

    Parameters
    ----------
    func : ty.ExternFunc
        The TIR function to be called.
    args : ty.TupleExpr
        The arguments to the TIR function.

    Returns
    -------
    ret : ty.Tensor
        The call node created
    """
    func = tg.check(0, "func", tg.ExternFunc(), func)
    args = tg.check(1, "args", tg.TupleExpr(), args)
    _ffi_func = _ffi.get_global_func("relax.op.vm.call_tir_dyn")
    return _ffi_func(func, args)


## (TVM-TOOL) py_op end vm/*

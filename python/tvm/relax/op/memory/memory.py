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
"""Relax memory primitives."""

from typing import Union
from . import _ffi_api
from ...expr import Expr, Call, PrimValue, DataTypeImm, StringImm
from ...utils import args_converter


@args_converter.auto
def alloc_storage(
    size: Expr,
    virtual_device_index: Union[int, Expr],
    storage_scope: Union[str, Expr],
    dtype: Union[str, Expr],
) -> Call:
    """Construct a Call to allocate a storage with specific size, virtual_device_index,
    storage_scope and dtype.

    Parameters
    ----------
    size : Expr
        The size of the storage to be allocated.

    virtual_device_index : Union[int, Expr]
        The virtual device index indicating on which device the storage is to be allocated.
        Index -1 is reserved for the host device.

    storage_scope : Union[str, Expr]
        The storage scope to allocate the storage to.

    dtype : Union[str, Expr]
        The datatype of the storage to be allocated.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated storage.
    """
    if isinstance(dtype, str):
        dtype = DataTypeImm(dtype)
    if isinstance(storage_scope, str):
        storage_scope = StringImm(storage_scope)
    if isinstance(virtual_device_index, int):
        virtual_device_index = PrimValue(virtual_device_index)
    return _ffi_api.alloc_storage(size, virtual_device_index, storage_scope, dtype)  # type: ignore


@args_converter.auto
def alloc_tensor(
    storage: Expr, offset: Union[int, Expr], shape: Expr, dtype: Union[str, Expr]
) -> Call:
    """Construct a Call to allocate a tensor on a certain storage starting from the given offset.

    Parameters
    ----------
    storage : Expr
        The storage to allocate the tensor to.

    offset : Union[int, Expr]
        The storage offset to allocate the tensor.

    shape : Expr
        The shape of the tensor to be allocated.

    dtype : Union[str, Expr]
        The datatype of the tensor to be allocated.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated tensor.
    """
    if isinstance(offset, int):
        offset = PrimValue(offset)
    if isinstance(dtype, str):
        dtype = DataTypeImm(dtype)
    return _ffi_api.alloc_tensor(storage, offset, shape, dtype)  # type: ignore


@args_converter.auto
def kill_storage(storage: Expr) -> Call:
    """Construct a Call to kill a storage.

    Parameters
    ----------
    storage : Expr
        The storage to be killed.

    Returns
    -------
    result : Call
        A relax Call to kill a storage.
    """
    return _ffi_api.kill_storage(storage)  # type: ignore


@args_converter.auto
def kill_tensor(tensor: Expr) -> Call:
    """Construct a Call to kill a tensor.

    Parameters
    ----------
    tensor : Expr
        The tensor to be killed.

    Returns
    -------
    result : Call
        A relax Call to kill a tensor.
    """
    return _ffi_api.kill_tensor(tensor)  # type: ignore

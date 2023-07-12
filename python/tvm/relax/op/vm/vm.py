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
"""Relax vm primitives."""

from typing import Union
from . import _ffi_api
from ...expr import Expr, Call, PrimValue, DataTypeImm, Tuple, StringImm
from ...utils import args_converter


@args_converter.auto
def alloc_storage(
    shape: Expr,
    runtime_device_index: Union[int, Expr],
    dtype: Union[str, Expr],
    storage_scope: Union[str, StringImm] = "global",
) -> Call:
    """Construct a Call to allocate a storage with specific size,
    runtime_device_index, and dtype.

    Parameters
    ----------
    shape : Expr
        The shape of the storage to be allocated.

    runtime_device_index : Union[int, Expr]
        The device index indicating on which device the tensor is to
        be allocated at runtime. Index -1 is reserved for the host device.

    dtype : Union[str, Expr]
        The datatype of the storage to be allocated.

    storage_scope : Union[str, StringImm]
        The storage scope of the storage to allocate. Default is global.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated storage.
    """
    if isinstance(dtype, str):
        dtype = DataTypeImm(dtype)
    if isinstance(storage_scope, str):
        storage_scope = StringImm(storage_scope)
    if isinstance(runtime_device_index, int):
        runtime_device_index = PrimValue(runtime_device_index)
    return _ffi_api.alloc_storage(shape, runtime_device_index, dtype, storage_scope)  # type: ignore


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


def kill_object(obj: Expr) -> Call:
    """Construct a Call to set the register corresponding to the input object to
    null at runtime, in order to kill the input object.

    Parameters
    ----------
    obj : Expr
        The object to be killed.

    Returns
    -------
    result : Call
        CallNode that kills the input object.
    """
    return _ffi_api.kill_object(obj)  # type: ignore


@args_converter.auto
def call_tir_dyn(func: Expr, args: Tuple) -> Call:
    """Construct a Call to call_tir_dyn (invoke the given TIR PrimFunc)
    consisting of the input tensors and the shape of the result.

    Parameters
    ----------
    func : Expr
        An expression evaluating to a TIR PrimFunc.

    args : Tuple
        The input args, includes a list of tensors, and a ShapeExpr.

    Returns
    -------
    result : Call
        A relax Call to call_tir_dyn.
    """
    if isinstance(args, (list, tuple)):
        args = Tuple(args)

    return _ffi_api.call_tir_dyn(func, args)  # type: ignore

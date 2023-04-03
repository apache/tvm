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


## (TVM-TOOL) py_op begin builtin/*
def alloc_tensor(
    shape: ty.Shape,
    dtype: ty.DType,
    runtime_device_index: ty.Int,
) -> Call:
    """Construct a Call to allocate a tensor with specific shape, dtype, and the index of
        the device it is constructed on.

    Parameters
    ----------
    shape : ty.Shape
        The shape of the tensor.
    dtype : ty.DType
        The data type of the tensor.
    runtime_device_index : ty.Int
        The index of the device it is constructed on.

    Returns
    -------
    ret : ty.Tensor
        The created call node.
    """
    shape = tg.check(0, "shape", tg.Shape(), shape)
    dtype = tg.check(1, "dtype", tg.DType(), dtype)
    runtime_device_index = tg.check(
        2, "runtime_device_index", tg.Int(), runtime_device_index
    )
    _ffi_func = _ffi.get_global_func("relax.op.builtin.alloc_tensor")
    return _ffi_func(shape, dtype, runtime_device_index)


def stop_lift_params(
    x: ty.Tensor,
) -> Call:
    """An indicator op that the consumers of input tensor should not be
        lifted to transform_params function.

    Parameters
    ----------
    x : ty.Tensor
        The input tensor.

    Returns
    -------
    ret : ty.Tensor
        The created call node.
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    _ffi_func = _ffi.get_global_func("relax.op.builtin.stop_lift_params")
    return _ffi_func(x)


## (TVM-TOOL) py_op end builtin/*

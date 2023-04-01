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
# pylint: disable=redefined-builtin
"""Linear algebra operators."""
from tvm import _ffi

from ..expr import Call
from . import ty
from . import ty_guard as tg

# pylint: disable=invalid-name


## (TVM-TOOL) py_op begin create/*
def full(
    shape: ty.Shape,
    fill_value: ty.Union[ty.PrimExpr, ty.Tensor],
    dtype: ty.DType = None,
) -> Call:
    """TBD

    Parameters
    ----------
    shape : ty.Shape
        The shape of the output tensor.
    fill_value : ty.Union[ty.PrimExpr, ty.Tensor]
        The value to fill the output tensor with.
    dtype : ty.DType
        The data type of the output tensor.

    Returns
    -------
    ret : ty.Tensor
        The output tensor.
    """
    shape = tg.check(0, "shape", tg.Shape(), shape)
    fill_value = tg.check(
        1, "fill_value", tg.Union(tg.PrimExpr(), tg.Tensor([0])), fill_value
    )
    dtype = tg.check(2, "dtype", tg.DType(), dtype)
    _ffi_func = _ffi.get_global_func("relax.op.full")
    return _ffi_func(shape, fill_value, dtype)


def full_like(
    x: ty.Tensor,
    fill_value: ty.Union[ty.PrimExpr, ty.Tensor],
    dtype: ty.DType = None,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    fill_value : ty.Union[ty.PrimExpr, ty.Tensor]
        TODO(tvm-unity-team): add doc
    dtype : ty.DType
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    fill_value = tg.check(
        1, "fill_value", tg.Union(tg.PrimExpr(), tg.Tensor([0])), fill_value
    )
    dtype = tg.check(2, "dtype", tg.DType(), dtype)
    _ffi_func = _ffi.get_global_func("relax.op.full_like")
    return _ffi_func(x, fill_value, dtype)


def ones(
    shape: ty.Shape,
    dtype: ty.DType = None,
) -> Call:
    """TBD

    Parameters
    ----------
    shape : ty.Shape
        TODO(tvm-unity-team): add doc
    dtype : ty.DType
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    shape = tg.check(0, "shape", tg.Shape(), shape)
    dtype = tg.check(1, "dtype", tg.DType(), dtype)
    _ffi_func = _ffi.get_global_func("relax.op.ones")
    return _ffi_func(shape, dtype)


def ones_like(
    x: ty.Tensor,
    dtype: ty.DType = None,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    dtype : ty.DType
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    dtype = tg.check(1, "dtype", tg.DType(), dtype)
    _ffi_func = _ffi.get_global_func("relax.op.ones_like")
    return _ffi_func(x, dtype)


def tril(
    x: ty.Tensor,
    k: ty.IntPrimExpr = 0,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    k : ty.IntPrimExpr
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    k = tg.check(1, "k", tg.IntPrimExpr(), k)
    _ffi_func = _ffi.get_global_func("relax.op.tril")
    return _ffi_func(x, k)


def triu(
    x: ty.Tensor,
    k: ty.IntPrimExpr = 0,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    k : ty.IntPrimExpr
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    k = tg.check(1, "k", tg.IntPrimExpr(), k)
    _ffi_func = _ffi.get_global_func("relax.op.triu")
    return _ffi_func(x, k)


def zeros(
    shape: ty.Shape,
    dtype: ty.DType = None,
) -> Call:
    """TBD

    Parameters
    ----------
    shape : ty.Shape
        TODO(tvm-unity-team): add doc
    dtype : ty.DType
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    shape = tg.check(0, "shape", tg.Shape(), shape)
    dtype = tg.check(1, "dtype", tg.DType(), dtype)
    _ffi_func = _ffi.get_global_func("relax.op.zeros")
    return _ffi_func(shape, dtype)


def zeros_like(
    x: ty.Tensor,
    dtype: ty.DType = None,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    dtype : ty.DType
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    dtype = tg.check(1, "dtype", tg.DType(), dtype)
    _ffi_func = _ffi.get_global_func("relax.op.zeros_like")
    return _ffi_func(x, dtype)


## (TVM-TOOL) py_op end create/*

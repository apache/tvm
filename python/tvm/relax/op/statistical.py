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
"""Statistical operators."""
from tvm import _ffi

from ..expr import Call
from . import ty
from . import ty_guard as tg

# pylint: disable=invalid-name


## (TVM-TOOL) py_op begin statistical/*
def max(
    x: ty.Tensor,
    axis: ty.Axes = None,
    keepdims: ty.Bool = False,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    axis : ty.Axes
        TODO(tvm-unity-team): add doc
    keepdims : ty.Bool
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    axis = tg.check(1, "axis", tg.Axes(x, False, True), axis)
    keepdims = tg.check(2, "keepdims", tg.Bool(), keepdims)
    _ffi_func = _ffi.get_global_func("relax.op.max")
    return _ffi_func(x, axis, keepdims)


def mean(
    x: ty.Tensor,
    axis: ty.Axes = None,
    keepdims: ty.Bool = False,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    axis : ty.Axes
        TODO(tvm-unity-team): add doc
    keepdims : ty.Bool
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    axis = tg.check(1, "axis", tg.Axes(x, False, True), axis)
    keepdims = tg.check(2, "keepdims", tg.Bool(), keepdims)
    _ffi_func = _ffi.get_global_func("relax.op.mean")
    return _ffi_func(x, axis, keepdims)


def min(
    x: ty.Tensor,
    axis: ty.Axes = None,
    keepdims: ty.Bool = False,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    axis : ty.Axes
        TODO(tvm-unity-team): add doc
    keepdims : ty.Bool
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    axis = tg.check(1, "axis", tg.Axes(x, False, True), axis)
    keepdims = tg.check(2, "keepdims", tg.Bool(), keepdims)
    _ffi_func = _ffi.get_global_func("relax.op.min")
    return _ffi_func(x, axis, keepdims)


def prod(
    x: ty.Tensor,
    axis: ty.Axes = None,
    keepdims: ty.Bool = False,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    axis : ty.Axes
        TODO(tvm-unity-team): add doc
    keepdims : ty.Bool
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    axis = tg.check(1, "axis", tg.Axes(x, False, True), axis)
    keepdims = tg.check(2, "keepdims", tg.Bool(), keepdims)
    _ffi_func = _ffi.get_global_func("relax.op.prod")
    return _ffi_func(x, axis, keepdims)


def std(
    x: ty.Tensor,
    axis: ty.Axes = None,
    keepdims: ty.Bool = False,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    axis : ty.Axes
        TODO(tvm-unity-team): add doc
    keepdims : ty.Bool
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    axis = tg.check(1, "axis", tg.Axes(x, False, True), axis)
    keepdims = tg.check(2, "keepdims", tg.Bool(), keepdims)
    _ffi_func = _ffi.get_global_func("relax.op.std")
    return _ffi_func(x, axis, keepdims)


def sum(
    x: ty.Tensor,
    axis: ty.Axes = None,
    keepdims: ty.Bool = False,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    axis : ty.Axes
        TODO(tvm-unity-team): add doc
    keepdims : ty.Bool
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    axis = tg.check(1, "axis", tg.Axes(x, False, True), axis)
    keepdims = tg.check(2, "keepdims", tg.Bool(), keepdims)
    _ffi_func = _ffi.get_global_func("relax.op.sum")
    return _ffi_func(x, axis, keepdims)


def variance(
    x: ty.Tensor,
    axis: ty.Axes = None,
    keepdims: ty.Bool = False,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    axis : ty.Axes
        TODO(tvm-unity-team): add doc
    keepdims : ty.Bool
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    axis = tg.check(1, "axis", tg.Axes(x, False, True), axis)
    keepdims = tg.check(2, "keepdims", tg.Bool(), keepdims)
    _ffi_func = _ffi.get_global_func("relax.op.variance")
    return _ffi_func(x, axis, keepdims)


## (TVM-TOOL) py_op end statistical/*

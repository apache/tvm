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
"""Image operators."""
from tvm import _ffi
from tvm.tir import IndexMap

from ..expr import Call
from . import ty
from . import ty_guard as tg

# pylint: disable=invalid-name,too-many-arguments


## (TVM-TOOL) py_op begin manipulate/*
def broadcast_to(
    x: ty.Tensor,
    shape: ty.Shape,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    shape : ty.Shape
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    shape = tg.check(1, "shape", tg.Shape(), shape)
    _ffi_func = _ffi.get_global_func("relax.op.broadcast_to")
    return _ffi_func(x, shape)


def concat(
    x: ty.Array[ty.Tensor],
    axis: ty.Axis = 0,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Array[ty.Tensor]
        TODO(tvm-unity-team): add doc
    axis : ty.Axis
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Array(tg.Tensor([]), []), x)
    axis = tg.check(1, "axis", tg.Axis(x, False, True), axis)
    _ffi_func = _ffi.get_global_func("relax.op.concat")
    return _ffi_func(x, axis)


def expand_dims(
    x: ty.Tensor,
    axis: ty.Axes,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    axis : ty.Axes
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    axis = tg.check(1, "axis", tg.Axes(x, True, False), axis)
    _ffi_func = _ffi.get_global_func("relax.op.expand_dims")
    return _ffi_func(x, axis)


def flatten(
    x: ty.Tensor,
    start_dim: ty.Axis = 0,
    end_dim: ty.Axis = -1,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    start_dim : ty.Axis
        TODO(tvm-unity-team): add doc
    end_dim : ty.Axis
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    start_dim = tg.check(1, "start_dim", tg.Axis(x, False, True), start_dim)
    end_dim = tg.check(2, "end_dim", tg.Axis(x, False, True), end_dim)
    _ffi_func = _ffi.get_global_func("relax.op.flatten")
    return _ffi_func(x, start_dim, end_dim)


def layout_transform(
    x: ty.Tensor,
    index_map: ty.IndexMap,
    pad_value: ty.Optional[ty.Float] = None,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    index_map : ty.IndexMap
        TODO(tvm-unity-team): add doc
    pad_value : ty.Optional[ty.Float]
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    index_map = tg.check(1, "index_map", tg.IndexMap(), index_map)
    pad_value = tg.check(2, "pad_value", tg.Optional(tg.Float()), pad_value)
    _ffi_func = _ffi.get_global_func("relax.op.layout_transform")
    return _ffi_func(x, index_map, pad_value)


def permute_dims(
    x: ty.Tensor,
    axes: ty.Optional[ty.Axes] = None,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    axes : ty.Optional[ty.Axes]
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    axes = tg.check(1, "axes", tg.Optional(tg.Axes(x, False, False)), axes)
    _ffi_func = _ffi.get_global_func("relax.op.permute_dims")
    return _ffi_func(x, axes)


def repeat(
    x: ty.Tensor,
    repeats: ty.Array[ty.IntPrimExpr],
    axis: ty.Optional[ty.Axis] = None,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    repeats : ty.Array[ty.IntPrimExpr]
        TODO(tvm-unity-team): add doc
    axis : ty.Optional[ty.Axis]
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    repeats = tg.check(1, "repeats", tg.Array(tg.IntPrimExpr(), []), repeats)
    axis = tg.check(2, "axis", tg.Optional(tg.Axis(x, False, True)), axis)
    _ffi_func = _ffi.get_global_func("relax.op.repeat")
    return _ffi_func(x, repeats, axis)


def reshape(
    x: ty.Tensor,
    shape: ty.Shape,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    shape : ty.Shape
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    shape = tg.check(1, "shape", tg.Shape(), shape)
    _ffi_func = _ffi.get_global_func("relax.op.reshape")
    return _ffi_func(x, shape)


def split(
    x: ty.Tensor,
    indices_or_sections: ty.Union[ty.Int, ty.Array[ty.IntPrimExpr]],
    axis: ty.Axis = 0,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    indices_or_sections : ty.Union[ty.Int, ty.Array[ty.IntPrimExpr]]
        TODO(tvm-unity-team): add doc
    axis : ty.Axis
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Array[ty.Tensor]
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    indices_or_sections = tg.check(
        1,
        "indices_or_sections",
        tg.Union(tg.Int(), tg.Array(tg.IntPrimExpr(), [], restrict=True)),
        indices_or_sections,
    )
    axis = tg.check(2, "axis", tg.Axis(x, False, True), axis)
    _ffi_func = _ffi.get_global_func("relax.op.split")
    return _ffi_func(x, indices_or_sections, axis)


def squeeze(
    x: ty.Tensor,
    axis: ty.Optional[ty.Axes] = None,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    axis : ty.Optional[ty.Axes]
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    axis = tg.check(1, "axis", tg.Optional(tg.Axes(x, False, True)), axis)
    _ffi_func = _ffi.get_global_func("relax.op.squeeze")
    return _ffi_func(x, axis)


def tile(
    x: ty.Tensor,
    repeats: ty.Shape,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    repeats : ty.Shape
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    repeats = tg.check(1, "repeats", tg.Shape(), repeats)
    _ffi_func = _ffi.get_global_func("relax.op.tile")
    return _ffi_func(x, repeats)


## (TVM-TOOL) py_op end manipulate/*

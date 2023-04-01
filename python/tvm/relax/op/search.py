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
"""Search operators."""
from tvm import _ffi

from ..expr import Call
from . import ty
from . import ty_guard as tg

# pylint: disable=invalid-name


## (TVM-TOOL) py_op begin search/*
def argmax(
    a: ty.Tensor,
    axis: ty.Axes = None,
    keepdims: ty.Bool = False,
) -> Call:
    """TBD

    Parameters
    ----------
    a : ty.Tensor
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
    a = tg.check(0, "a", tg.Tensor([]), a)
    axis = tg.check(1, "axis", tg.Axes(a, False, True), axis)
    keepdims = tg.check(2, "keepdims", tg.Bool(), keepdims)
    _ffi_func = _ffi.get_global_func("relax.op.argmax")
    return _ffi_func(a, axis, keepdims)


def argmin(
    a: ty.Tensor,
    axis: ty.Axes = None,
    keepdims: ty.Bool = False,
) -> Call:
    """TBD

    Parameters
    ----------
    a : ty.Tensor
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
    a = tg.check(0, "a", tg.Tensor([]), a)
    axis = tg.check(1, "axis", tg.Axes(a, False, True), axis)
    keepdims = tg.check(2, "keepdims", tg.Bool(), keepdims)
    _ffi_func = _ffi.get_global_func("relax.op.argmin")
    return _ffi_func(a, axis, keepdims)


def where(
    condition: ty.BoolTensor,
    x: ty.Tensor,
    y: ty.Tensor,
) -> Call:
    """TBD

    Parameters
    ----------
    condition : ty.BoolTensor
        TODO(tvm-unity-team): add doc
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    y : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    condition = tg.check(0, "condition", tg.BoolTensor([]), condition)
    x = tg.check(1, "x", tg.Tensor([]), x)
    y = tg.check(2, "y", tg.Tensor([]), y)
    _ffi_func = _ffi.get_global_func("relax.op.where")
    return _ffi_func(condition, x, y)


## (TVM-TOOL) py_op end search/*

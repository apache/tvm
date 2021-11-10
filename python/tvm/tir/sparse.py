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
"""SparseTIR axes and SparseBuffer
"""
from typing import List, Dict, Optional
import tvm._ffi
from tvm.ir import PrimExpr
from tvm.runtime import Object, const
from tvm.tir import Var

from . import _ffi_api
from .buffer import Buffer


class Axis(Object):
    """Base class of all the sparse axes."""

    @property
    def name(self):
        return _ffi_api.GetAxisName(self)

    @property
    def length(self):
        return _ffi_api.GetAxisLength(self)

    @property
    def idtype(self):
        return _ffi_api.GetAxisIndexType(self)


class DenseAxis(Axis):
    pass


class SparseAxis(Axis):
    pass


@tvm._ffi.register_object("tir.sparse.DenseFixedAxis")
class DenseFixedAxis(DenseAxis):
    """DenseFixedAxis node

    Parameters
    ----------
    name : str
        The name of the axis

    length : PrimExpr
        The length of the axis

    from_sparse : Optional[SparseAxis]
        The SparseAxis that this axis is created from
    """

    name: str
    length: PrimExpr
    from_sparse: Optional[SparseAxis]

    def __init__(self, name, length, from_sparse=None):
        self.__init_handle_by_constructor__(_ffi_api.DenseFixedAxis, name, length, from_sparse)  # type: ignore


@tvm._ffi.register_object("tir.sparse.DenseVariableAxis")
class DenseVariableAxis(DenseAxis):
    """DenseVariableAxis node

    Parameters
    ----------
    name : str
        The name of the axis

    length : PrimExpr
        The length of the axis

    indptr : Buffer
        The indptr buffer of the axis
    """

    name: str
    length: PrimExpr
    indptr: Buffer

    def __init__(self, name, length, indptr):
        self.__init_handle_by_constructor__(
            _ffi_api.DenseVariableAxis, name, length, indptr  # type: ignore
        )


@tvm._ffi.register_object("tir.sparse.SparseFixedAxis")
class SparseFixedAxis(DenseAxis):
    """SparseFixedAxis node

    Parameters
    ----------
    name : str
        The name of the axis

    length : PrimExpr
        The length of the axis

    indices : Buffer
        The indices buffer of the axis

    num_cols : PrimExpr
        The number of non-zero elements along the axis
    """

    name: str
    length: PrimExpr
    indices: Buffer
    num_cols: PrimExpr

    def __init__(self, name, length, indices, num_cols):
        self.__init_handle_by_constructor__(
            _ffi_api.SparseFixedAxis, name, length, indices, num_cols  # type: ignore
        )


@tvm._ffi.register_object("tir.sparse.SparseVariableAxis")
class SparseVariableAxis(DenseAxis):
    """SparseVariableAxis node

    Parameters
    ----------
    name : str
        The name of the axis

    length : PrimExpr
        The length of the axis

    indptr : Buffer
        The indptr buffer of the axis

    indices : Buffer
        The indices buffer of the axis
    """

    name: str
    length: PrimExpr
    indptr: Buffer
    indices: Buffer

    def __init__(self, name, length, indptr, indices):
        self.__init_handle_by_constructor__(
            _ffi_api.SparseVariableAxis, name, length, indptr, indices  # type: ignore
        )


@tvm._ffi.register_object("tir.sparse.AxisTree")
class AxisTree(Object):
    """AxisTree node

    Parameters
    ----------
    axis_parent_map: Dict
        A dictionary that maps axis name to parent axis name, value is None if there is not parent axis.
    """

    axis_parent_map: Dict[str, Optional[str]]

    def __init__(self, axis_parent_map) -> None:
        keys = list(axis_parent_map.keys())
        values = list(axis_parent_map.values())
        self.__init_handle_by_constructor__(
            _ffi_api.AxisTree, keys, values  # type:ignore
        )


@tvm._ffi.register_object("tir.sparse.SparseBuffer")
class SparseBuffer(Object):
    """SparseBuffer node

    Parameters
    ----------
    axes : List[Axis]
        The axes of the sparse buffer

    data : Buffer
        The data of the sparse buffer

    name : str
        The name of the sparse buffer
    """

    axes: List[Axis]
    data: Buffer
    name: str

    def __init__(self, axes, data, name):
        self.__init_handle_by_constructor__(_ffi_api.SparseBuffer, axes, data, name)  # type: ignore


@tvm._ffi.register_object("tir.sparse.SpIterVar")
class SpIterVar(Object):
    """IterVar in SparseTIR

    Parameters
    ----------
    var : Var
        The var of the SpIterVar

    max_extent : PrimExpr
        The maximum extent of the SpIterVar

    kind : int
        The kind of the SpIterVar

    is_reduction : bool
        Whether the SpIterVar is a reduction iterator

    axis : Axis
        The axis over which the SpIterVar iterates
    """

    var: Var
    max_extent: PrimExpr
    kind: int
    is_reduction: bool
    axis: Axis

    DenseFixed = 0
    DenseVariable = 1
    SparseFixed = 2
    SparseVariable = 3

    def __init__(self, var, max_extent, kind, is_reduction, axis):
        self.__init_handle_by_constructor__(
            _ffi_api.SpIterVar, var, max_extent, kind, is_reduction, axis  # type: ignore
        )

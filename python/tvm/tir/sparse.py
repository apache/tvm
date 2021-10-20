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
from typing import List
import tvm._ffi
from tvm.ir import PrimExpr
from tvm.runtime import Object, const

from . import _ffi_api
from .buffer import Buffer


class Axis(Object):
    """Base class of all the sparse axes."""


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
    """

    name: str
    length: PrimExpr

    def __init__(self, name, length):
        self.__init_handle_by_constructor__(
            _ffi_api.DenseFixedAxis, name, length  # type: ignore
        )


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
class AxisTree:
    # Todo(@ruihang): to do later
    pass


@tvm._ffi.register_object("tir.sparse.SparseBuffer")
class SparseBuffer:
    """SparseBuffer node

    Parameters
    ----------
    root : AxisTree
        The root of the axis dependency tree of the sparse buffer

    axes : List[Axis]
        The axes of the sparse buffer

    ndim : int
        The number of dimensions of the sparse buffer

    data : Buffer
        The data of the sparse buffer
    """

    root: AxisTree
    axes: List[Axis]
    ndim: int
    data: Buffer

    def __init__(self, root, axes, ndim, data):
        self.__init_handle_by_constructor__(
            _ffi_api.SparseBuffer, root, axes, ndim, data  # type: ignore
        )

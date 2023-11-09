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
# pylint: disable=redefined-builtin, wrong-import-order, no-member, invalid-name, unused-import

"""IRBuilder for distributed Relax dialect"""
from typing import Union, List, Tuple, Optional

import numpy as _np  # type: ignore
import tvm

from tvm.ir import PrimExpr
from tvm.relax.expr import Expr, ShapeExpr, Call, ExternFunc, Constant
from tvm.relax.expr import Tuple as RxTuple
from tvm.relax.distributed import DTensorStructInfo
from tvm.relax.utils import args_converter
from tvm._ffi import base as _base
from tvm.runtime import ndarray as _nd
from tvm.relax.op.distributed import (
    redistribute as _redistribute,
    annotate_sharding as _annotate_sharding,
    call_tir_local_view,
    redistribute_replica_to_shard,
)
from tvm.relax.distributed import DeviceMesh, Placement
from . import _ffi_api
from ..ir import py_str
from ...ir import IRModuleFrame
from ... import IRBuilder


@args_converter.auto
def call_tir(
    func: Union[str, Expr],
    args: Expr,
    out_sinfo: Union[DTensorStructInfo, List[DTensorStructInfo]],
    tir_vars: Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]] = None,
) -> Call:
    """Distributed version of call_tir

    Parameters:
    ----------
    func : Union[str, Expr]
        The destination-passing-style function, can be ExternFunc or PrimFunc.

    args : Expr
        The input arguments.

    out_sinfo : Union[DTensorStructInfo, List[DTensorStructInfo]]
        The structure info of the call_tir output.
        It should be a single or a list of DTensorStructInfo. Each one denotes the
        structure info of a returned distributed tensor.

    tir_vars : Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]]
        ShapeExpr representing a tuple of integers to unpack when calling func. Is null if not used

    Returns
    -------
    ret: Call
        A call node for the call_tir operator.
    """
    if isinstance(func, str):
        func = ExternFunc(func)

    if isinstance(args, Expr) and not isinstance(args, RxTuple):  # type: ignore
        args = RxTuple((args,))

    if not isinstance(out_sinfo, list):
        out_sinfo = [out_sinfo]

    if isinstance(tir_vars, (list, tuple)):
        tir_vars = ShapeExpr(tir_vars)

    return _ffi_api.call_tir_dist(func, args, out_sinfo, tir_vars)  # type: ignore


def const(
    value: Union[bool, int, float, _np.ndarray, tvm.nd.NDArray],
    struct_info: DTensorStructInfo,
) -> Constant:
    """Create a constant value.

    Parameters
    ----------
    value: Union[bool, int, float, numpy.ndarray, tvm.nd.NDArray]
        The constant value.

    dtype: Optional[str]
        The data type of the resulting constant.

    Note
    ----
    When dtype is None, we use the following rule:

    - int maps to "int32"
    - float maps to "float32"
    - bool maps to "bool"
    - other using the same default rule as numpy.
    """
    struct_info = tvm.runtime.convert_to_object(struct_info)
    if not isinstance(struct_info, DTensorStructInfo):
        raise TypeError("struct_info needs to be an instance of DTensorStructInfo. ")
    dtype = str(struct_info.tensor_sinfo.dtype)
    if isinstance(value, (_base.numeric_types, (bool, list))):
        value = _np.array(value, dtype=dtype)

    if isinstance(value, (_np.ndarray, _np.generic)):
        if dtype is not None:
            value = value.astype(dtype)
        value = _nd.array(value)

    if not isinstance(value, _nd.NDArray):
        raise ValueError("value has to be scalar or NDArray")

    return Constant(value, struct_info)


def _lookup_device_mesh(device_mesh_str: py_str) -> DeviceMesh:
    if not IRBuilder.is_in_scope():
        raise ValueError("device_mesh cannot be found in global info")
    name, index_str = device_mesh_str.split("[")
    index = int(index_str[:-1])
    frames = IRBuilder.current().frames
    for f in frames:
        if isinstance(f, IRModuleFrame):
            device_mesh = f.global_infos[name][index]
            break
    assert isinstance(device_mesh, DeviceMesh)
    return device_mesh


def annotate_sharding(
    value: Expr, device_mesh: Union[py_str, DeviceMesh], placement: Union[py_str, Placement]
) -> Expr:
    if isinstance(device_mesh, py_str):
        device_mesh = _lookup_device_mesh(device_mesh)
    if isinstance(placement, py_str):
        placement = Placement.from_text(placement)
    return _annotate_sharding(value, device_mesh, placement)


def redistribute(
    value: Expr, device_mesh: Union[py_str, DeviceMesh], placement: Union[py_str, Placement]
) -> Expr:
    if isinstance(device_mesh, py_str):
        device_mesh = _lookup_device_mesh(device_mesh)
    if isinstance(placement, py_str):
        placement = Placement.from_text(placement)
    return _redistribute(value, device_mesh, placement)

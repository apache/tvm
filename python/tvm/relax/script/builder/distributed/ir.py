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
# ruff: noqa: F401

"""IRBuilder for distributed Relax dialect"""

from numbers import Number
from typing import Optional, Union

import numpy as _np  # type: ignore

import tvm
from tvm import base as _base
from tvm.ir import Call, GenericConst
from tvm.relax.distributed import DeviceMesh, DTensorType, Placement
from tvm.relax.expr import Expr, ExternFunc
from tvm.relax.expr import Tuple as RxTuple
from tvm.relax.op.distributed import (
    annotate_sharding as _annotate_sharding,
)
from tvm.relax.op.distributed import (
    call_tir_local_view,
    redistribute_replica_to_shard,
)
from tvm.relax.op.distributed import (
    redistribute as _redistribute,
)
from tvm.relax.script.builder.ir import py_str
from tvm.relax.utils import convert_to_expr
from tvm.runtime import _tensor
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder.ir import IRModuleFrame
from tvm.script.ir_builder.ir.ir import lookup_global_info

from . import _ffi_api


def call_tir(
    func: str | Expr,
    args: Expr,
    out_ty: DTensorType | list[DTensorType],
) -> Call:
    """Distributed version of call_tir

    Parameters:
    ----------
    func : Union[str, Expr]
        The destination-passing-style function, can be ExternFunc or PrimFunc.

    args : Expr
        The ordered distributed-tensor and primitive input arguments.  These
        correspond positionally to the leading parameters of the PrimFunc.

    out_ty : Union[DTensorType, List[DTensorType]]
        The type information of the call_tir output.
        It should be a single or a list of DTensorType. Each one denotes the
        type information of a returned distributed tensor.

    Returns
    -------
    ret: Call
        A call node for the call_tir operator.
    """
    if isinstance(func, str):
        func = ExternFunc(func)

    if isinstance(args, tuple | list):
        args = RxTuple([convert_to_expr(a) for a in args])
    elif isinstance(args, Expr) and not isinstance(args, RxTuple):  # type: ignore
        args = RxTuple((args,))

    if not isinstance(out_ty, list):
        out_ty = [out_ty]

    return _ffi_api.call_tir_dist(func, args, out_ty)  # type: ignore


def const(
    value: bool | int | float | _np.ndarray | tvm.runtime.Tensor,
    ty: DTensorType,
) -> GenericConst:
    """Create a distributed constant value with the specified tensor type.

    Parameters
    ----------
    value : bool, int, float, numpy.ndarray or tvm.runtime.Tensor
        The constant value.

    ty : DTensorType
        The distributed tensor type, including its tensor dtype, device mesh
        and placement.

    Returns
    -------
    res : GenericConst
        The constant carrying the supplied distributed tensor type.

    Notes
    -----
    Python scalars and NumPy values are converted to the dtype specified by
    ``ty.tensor_ty``. The dtype is not inferred from the Python value.
    """
    ty = tvm.runtime.convert(ty)
    if not isinstance(ty, DTensorType):
        raise TypeError("ty needs to be an instance of DTensorType. ")
    dtype = str(ty.tensor_ty.dtype)
    if isinstance(value, Number | (bool | list)):
        value = _np.array(value, dtype=dtype)

    if isinstance(value, _np.ndarray | _np.generic):
        if dtype is not None:
            value = value.astype(dtype)
        value = _tensor.tensor(value)

    if not isinstance(value, _tensor.Tensor):
        raise ValueError("value has to be scalar or Tensor")

    return GenericConst(value, ty)


def _lookup_device_mesh(device_mesh_str: py_str) -> DeviceMesh:
    name, index_str = device_mesh_str.split("[")
    index = int(index_str[:-1])
    if not IRBuilder.is_in_scope():
        device_mesh = lookup_global_info(name, index)
        if not isinstance(device_mesh, DeviceMesh):
            raise TypeError("The device_mesh global info must be a DeviceMesh.")
        return device_mesh
    frames = IRBuilder.current().frames
    for f in frames:
        if isinstance(f, IRModuleFrame):
            device_mesh = f.global_infos[name][index]
            break
    assert isinstance(device_mesh, DeviceMesh)
    return device_mesh


def annotate_sharding(
    value: Expr, device_mesh: py_str | DeviceMesh, placement: py_str | Placement
) -> Expr:
    if isinstance(device_mesh, py_str):
        device_mesh = _lookup_device_mesh(device_mesh)
    if isinstance(placement, py_str):
        placement = Placement.from_text(placement)
    return _annotate_sharding(value, device_mesh, placement)


def redistribute(
    value: Expr, device_mesh: py_str | DeviceMesh, placement: py_str | Placement
) -> Expr:
    if isinstance(device_mesh, py_str):
        device_mesh = _lookup_device_mesh(device_mesh)
    if isinstance(placement, py_str):
        placement = Placement.from_text(placement)
    return _redistribute(value, device_mesh, placement)

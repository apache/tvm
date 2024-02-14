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
# pylint: disable=redefined-builtin
"""Operators for distributed Relax.
"""
from typing import Union, List, Tuple, Optional

from tvm.relax.distributed.struct_info import DeviceMesh, Placement
from tvm.ir import PrimExpr
from tvm.relax.utils import args_converter
from tvm.relax.distributed import DTensorStructInfo
from ...expr import Tuple as RxTuple
from . import _ffi_api
from ...expr import Expr, ShapeExpr, Call, GlobalVar


def annotate_sharding(input: Expr, device_mesh: DeviceMesh, placement: Placement) -> Expr:
    """Annotate sharding plan for tensor

    Parameters
    ----------
    input : relax.Expr
      The input tensor.
    device_mesh: DeviceMesh
      The device mesh of the sharding plan
    placement: Placement
      The placement of the sharding plan

    Returns
    -------
    result : relax.Expr
      The tensor unmodified.
    """
    return _ffi_api.annotate_sharding(input, device_mesh, placement)  # type: ignore


def redistribute(input: Expr, device_mesh: DeviceMesh, placement: Placement) -> Expr:
    """Redistribute tensor

    Parameters
    ----------
    input : relax.Expr
      The input tensor.
    device_mesh: DeviceMesh
      The device mesh after redistribution
    placement: Placement
      The placement after redistribution
    Returns
    -------
    result : relax.Expr
      The tensor after redistribution.
    """
    return _ffi_api.redistribute(input, device_mesh, placement)  # type: ignore


@args_converter.auto
def call_tir_local_view(
    gvar: GlobalVar,
    args: Expr,
    out_sinfo: Union[DTensorStructInfo, List[DTensorStructInfo]],
    tir_vars: Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]] = None,
) -> Call:
    """
    Call a tir.prim_func and return the output. The prim_func should be a worker-local function
    that is actually executed on each worker, instead of the unpartitioned function.
    The output of this operator is DTensor or a tuple of DTensors.

    Parameters
    ----------
    gvar : GlobalVar
        The GlobalVar referring to a tir PrimFunc.

    args : Expr
        The input arguments.

    out_sinfo : Union[DTensorStructInfo, List[DTensorStructInfo]]
        The structure info of the call_tir output.
        It should be a single or a list of DTensorStructInfo. Each one denotes the
        structure info of a returned tensor.

    tir_vars : Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]]
        ShapeExpr representing a tuple of integers to unpack when calling func. Is null if not used

    Returns
    -------
    ret: Call
        A call node for the call_tir_local_view operator.
    """
    if isinstance(args, Expr) and not isinstance(args, RxTuple):  # type: ignore
        args = RxTuple((args,))

    if not isinstance(out_sinfo, list):
        out_sinfo = [out_sinfo]

    if isinstance(tir_vars, (list, tuple)):
        tir_vars = ShapeExpr(tir_vars)

    return _ffi_api.call_tir_local_view(gvar, args, out_sinfo, tir_vars)  # type: ignore


def redistribute_replica_to_shard(input: Expr, num_workers: int, axis: int) -> Expr:
    """Slice tensor into several parts along one axis,
        and each worker takes one part.
        input.struct_info.shape[axis] % num_workers == 0 is required.
        Each worker must have an identical copy of the input.
        This is a specialized version of redistribute op.

    Parameters
    ----------
    input : relax.Expr
      The buffer to be sliced into equal parts.

    num_worker : int
      The number of workers, i.e. the number of parts the given buffer should be sliced into.

    axis : int
      The axis of the tensor to be sliced.

    Returns
    -------
    result : relax.Expr
      Sliced Tensor kept by each device.
    """
    return _ffi_api.redistribute_replica_to_shard(input, num_workers, axis)

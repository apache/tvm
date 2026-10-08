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
"""Operators for distributed Relax."""

from tvm.error import InternalError as _InternalError
from tvm.ir import Call
from tvm.ir.attrs import make_node
from tvm.relax import ShapeExpr, TupleType
from tvm.relax.distributed import DeviceMesh, DTensorType, Placement

from ...expr import Expr, GlobalVar
from ...expr import Tuple as RxTuple
from ...utils import convert_to_expr


def annotate_sharding(
    input: Expr,
    device_mesh: DeviceMesh,
    placement: Placement,
    *,
    ty=None,
    span=None,
) -> Expr:
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
    return Call(
        "relax.dist.annotate_sharding",
        [input],
        attrs=make_node(
            "relax.attrs.DistributionAttrs", device_mesh=device_mesh, placement=placement
        ),
        ty=ty,
        span=span,
    )  # type: ignore


def redistribute(
    input: Expr,
    device_mesh: DeviceMesh,
    placement: Placement,
    *,
    ty=None,
    span=None,
) -> Expr:
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
    return Call(
        "relax.dist.redistribute",
        [input],
        attrs=make_node(
            "relax.attrs.DistributionAttrs", device_mesh=device_mesh, placement=placement
        ),
        ty=ty,
        span=span,
    )  # type: ignore


def call_tir_local_view(
    gvar: GlobalVar,
    args: Expr,
    out_ty: DTensorType | list[DTensorType] | None = None,
    *,
    ty_args=None,
    ty=None,
    span=None,
) -> Call:
    """
    Call a tirx.function and return the output. The function should be a worker-local function
    that is actually executed on each worker, instead of the unpartitioned function.
    The output of this operator is DTensor or a tuple of DTensors.

    Parameters
    ----------
    gvar : GlobalVar
        The GlobalVar referring to a tirx Function.

    args : Expr
        The ordered distributed-tensor and primitive input arguments.  These
        correspond positionally to the leading parameters of the Function.

    out_ty : Union[DTensorType, List[DTensorType]]
        The type information of the call_tir output.
        It should be a single or a list of DTensorType. Each one denotes the
        type information of a returned tensor.

    Returns
    -------
    ret: Call
        A call node for the call_tir_local_view operator.
    """
    if isinstance(args, tuple | list):
        args = RxTuple([convert_to_expr(a) for a in args])
    elif isinstance(args, Expr) and not isinstance(args, RxTuple):  # type: ignore
        args = RxTuple((args,))

    if out_ty is not None:
        if ty_args is not None:
            raise TypeError("Specify either out_ty or ty_args")
        if not isinstance(out_ty, list):
            out_ty = [out_ty]
        for output in out_ty:
            if not isinstance(output.tensor_ty.shape, ShapeExpr):
                raise _InternalError("out_ty must have a defined ShapeExpr shape")
        ty_args = [out_ty[0] if len(out_ty) == 1 else TupleType(out_ty)]
    return Call(
        "relax.dist.call_tir_local_view",
        [gvar, args],
        ty_args=ty_args,
        ty=ty,
        span=span,
    )


def redistribute_replica_to_shard(
    input: Expr, num_workers: int, axis: int, *, ty=None, span=None
) -> Expr:
    """Slice tensor into several parts along one axis,
        and each worker takes one part.
        input.ty.shape[axis] % num_workers == 0 is required.
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
    return Call(
        "relax.dist.redistribute_replica_to_shard",
        [input],
        attrs=make_node("relax.attrs.ScatterCollectiveAttrs", num_workers=num_workers, axis=axis),
        ty=ty,
        span=span,
    )

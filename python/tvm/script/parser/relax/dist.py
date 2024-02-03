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
# pylint: disable=redefined-builtin,missing-docstring, invalid-name, unused-import, redefined-outer-name

from typing import Any
from typing import Dict, List, Optional, Set, Union

from tvm.relax import TensorStructInfo
from tvm.ir import Range
from tvm.tir import PrimExpr
from tvm.relax.distributed import DeviceMesh, Placement, DTensorStructInfo, device_mesh
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder.ir import IRModuleFrame
from tvm.script.ir_builder.relax.distributed import (
    call_tir,
    const,
    annotate_sharding,
    redistribute,
    redistribute_replica_to_shard,
    call_tir_local_view,
)
from .entry import StructInfoProxy, TensorProxy


############################### R.DTensor ###############################


class DTensorProxy(StructInfoProxy):
    tensor_sinfo_proxy: TensorProxy
    device_mesh: DeviceMesh
    placement: Placement

    def __init__(
        self,
        tensor_sinfo_proxy: TensorProxy,
        device_mesh: DeviceMesh,
        placement: Placement,
    ) -> None:
        self.device_mesh = device_mesh
        self.placement = placement
        self.tensor_sinfo_proxy = tensor_sinfo_proxy
        super().__init__()

    def get_symbolic_vars(self) -> Set[str]:
        return self.tensor_sinfo_proxy.get_symbolic_vars()

    def as_struct_info(self, dict_globals: Optional[Dict[str, Any]] = None) -> TensorStructInfo:
        return DTensorStructInfo(
            self.tensor_sinfo_proxy.as_struct_info(dict_globals),
            self.device_mesh,
            self.placement,
        )


def DTensor(
    shape: Optional[List[Union[PrimExpr, str]]] = None,
    dtype: Optional[str] = None,
    device_mesh: Union[DeviceMesh, str] = DeviceMesh([], Range(0, 1)),
    placement: Union[Placement, str] = "",
    *,
    ndim: int = -1,
) -> DTensorProxy:
    # scalar tensor case
    if shape is not None and len(shape) == 0:
        shape = []
    if isinstance(shape, str) and dtype is None:
        dtype = shape
        shape = None

    if shape is not None and not isinstance(shape, (tuple, list)):
        raise ValueError(f"shape must be a list or tuple, but got: {shape}")
    if isinstance(device_mesh, str):
        if not IRBuilder.is_in_scope():
            return (
                DTensorProxy(
                    TensorProxy(shape, dtype, None, ndim), DeviceMesh([], Range(0, 1)), ""
                ),
            )
        name, index = device_mesh.split("[")
        index = int(index[:-1])
        frames = IRBuilder.current().frames
        for f in frames:
            if isinstance(f, IRModuleFrame):
                device_mesh = f.global_infos[name][index]
                break
        assert isinstance(device_mesh, DeviceMesh)
    if isinstance(placement, str):
        placement = Placement.from_text(placement)
    return DTensorProxy(TensorProxy(shape, dtype, None, ndim), device_mesh, placement)


__all__ = ["DTensor", "device_mesh"]

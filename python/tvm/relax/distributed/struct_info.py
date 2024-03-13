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
# pylint: disable=redefined-builtin, invalid-name
"""Struct Info for distributed tensor."""
import enum
from typing import List
import tvm
from tvm.relax.struct_info import StructInfo, TensorStructInfo
from tvm.ir import Span
from tvm.runtime.object import Object
from tvm import TVMError

from .global_info import DeviceMesh
from . import _ffi_api


class PlacementSpecKind(enum.IntEnum):
    kSharding = 0
    kReplica = 1


@tvm._ffi.register_object("relax.distributed.PlacementSpec")
class PlacementSpec(Object):
    """Describes how data is distributed in one dimension of the device mesh

    Parameters
    ----------
    axis: int
        If the kind is sharding, this value represents the tensor dimension to shard.
        otherwise, axis is -1
    kind: PlacementSpecKind
        The kind of placement spec. Possible values: kSharding and kReplica.
    """

    axis: int
    kind: PlacementSpecKind

    def __init__(self, *args, **kwargs):
        raise TVMError("PlacementSpec is not intended to be constructed directly, ")

    @staticmethod
    def sharding(axis: int) -> "PlacementSpec":
        """Create a sharding placement spec

        Parameters
        ----------
        axis: int
            The tensor dimension to shard.

        Returns
        -------
        placement_spec: PlacementSpec
            The placement spec.
        """
        return _ffi_api.Sharding(axis)

    @staticmethod
    def replica() -> "PlacementSpec":
        """Create a replica placement spec

        Returns
        -------
        placement_spec: PlacementSpec
            The placement spec.
        """
        return _ffi_api.Replica()


@tvm._ffi.register_object("relax.distributed.Placement")
class Placement(Object):
    """Describes how data is distributed in each dimension of the device mesh

    Parameters
    ----------
    dim_specs: List[PlacementSpec]
        The placement spec for each dimension of the device mesh.
    """

    def __init__(self, dim_specs: List[PlacementSpec]):
        self.__init_handle_by_constructor__(_ffi_api.Placement, dim_specs)  # type: ignore

    @staticmethod
    def from_text(text: str) -> "Placement":
        """Create a placement from a text string.

        Parameters
        ----------
        text: str
            The text string.

        Returns
        -------
        placement: Placement
            The placement.
        """
        return _ffi_api.PlacementFromText(text)


@tvm._ffi.register_object("relax.DTensorStructInfo")
class DTensorStructInfo(StructInfo):
    """StructInfo of a Distributed Tensor value.

    Parameters
    ----------
    tensor_sinfo: TensorStructInfo
        The struct info inherited from TensorStructInfo
    device_mesh: DeviceMesh
        The device mesh of the tensor.
    placement: Placement
        The placement of the tensor among the device mesh

    """

    tensor_sinfo: TensorStructInfo
    device_mesh: DeviceMesh
    placement: Placement

    def __init__(
        self,
        tensor_sinfo: TensorStructInfo,
        device_mesh: DeviceMesh,
        placement: Placement,
        span: Span = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.DTensorStructInfo, tensor_sinfo, device_mesh, placement, span  # type: ignore
        )

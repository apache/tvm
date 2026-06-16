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
"""CUDA-specific TIRx language helpers."""

from importlib import import_module

__all__ = [
    "BaseTileScheduler",
    "ClusterPersistentScheduler2D",
    "FlashAttentionLPTScheduler",
    "FlashAttentionLinearScheduler",
    "GroupMajor3D",
    "IndexedTripleTileScheduler",
    "MBarrier",
    "Pipeline",
    "PipelineState",
    "RankAwareGroupMajorTileScheduler",
    "SMEMPool",
    "SmemDescriptor",
    "TCGen05Bar",
    "TMABar",
    "TMEMPool",
    "TMEMStages",
    "WarpRole",
    "WarpgroupRole",
]

_HELPER_MODULES = {
    "MBarrier": ".pipeline",
    "Pipeline": ".pipeline",
    "PipelineState": ".pipeline",
    "BaseTileScheduler": ".tile_scheduler",
    "ClusterPersistentScheduler2D": ".tile_scheduler",
    "FlashAttentionLPTScheduler": ".tile_scheduler",
    "FlashAttentionLinearScheduler": ".tile_scheduler",
    "GroupMajor3D": ".tile_scheduler",
    "IndexedTripleTileScheduler": ".tile_scheduler",
    "RankAwareGroupMajorTileScheduler": ".tile_scheduler",
    "SMEMPool": ".alloc_pool",
    "TCGen05Bar": ".pipeline",
    "TMABar": ".pipeline",
    "TMEMPool": ".alloc_pool",
    "TMEMStages": ".alloc_pool",
    "WarpRole": ".warp_role",
    "WarpgroupRole": ".warp_role",
    "SmemDescriptor": ".smem_desc",
}


def __getattr__(name: str):
    module_name = _HELPER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

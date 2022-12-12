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
"""Multi-level tiling with reuse."""
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Callable

from tvm.tir.schedule import Schedule, BlockRV
from tvm._ffi import register_object

from .. import _ffi_api
from .schedule_rule import ScheduleRule


class ReuseType(NamedTuple):
    """Reuse type."""

    req: str
    levels: List[int]
    scope: str

    def as_dict(self) -> Dict[str, Any]:
        """Return the dict representation of the reuse type."""
        return {
            "req": self.req,
            "levels": self.levels,
            "scope": self.scope,
        }


@register_object("meta_schedule.MultiLevelTiling")
class MultiLevelTiling(ScheduleRule):
    """Multi-level tiling with reuse.

    Parameters
    ----------
    structure : str
        The tiling structure. Recommended:
        - 'SSRSRS' on CPU
        - 'SSSRRSRS' on GPU
    tile_bind : Optional[List[str]]
        For each level of tiles, which thread axis it is bound to. Recommended:
        - None on CPU
        - [blockIdx.x, vthread.x, threadIdx.x] on GPU
    max_innermost_factor : Optional[int]
        The maximum size of the innermost factor. None means no limit
    vector_load_lens : Optional[List[int]]
        The length of vector lane in vectorized cooperative fetching.
        None means disable vectorization
    reuse_read : Optional[ReuseType]
        Data reuse configuration for reading. None means no reuse.
    reuse_write : Optional[ReuseType]
        Data reuse configuration for writing. None means no reuse.
    filter_fn: Optional[Callable[[Schedule, BlockRV], bool]]
        A function that can be passed to overwrite the default condition for applying
        MultiLevelTiling to a block. This is useful if there is a need to apply MultiLevelTiling
        to an operation / block which is ignored by default. This function should return True
        for a block that should be tiled (based on the block name, for example).
    """

    def __init__(
        self,
        structure: str,
        tile_binds: Optional[List[str]] = None,
        max_innermost_factor: Optional[int] = None,
        vector_load_lens: Optional[List[int]] = None,
        reuse_read: Optional[ReuseType] = None,
        reuse_write: Optional[ReuseType] = None,
        filter_fn: Optional[Callable[[Schedule, BlockRV], bool]] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRuleMultiLevelTiling,  # type: ignore # pylint: disable=no-member
            structure,
            tile_binds,
            max_innermost_factor,
            vector_load_lens,
            reuse_read.as_dict() if reuse_read is not None else None,
            reuse_write.as_dict() if reuse_write is not None else None,
            filter_fn,
        )


@register_object("meta_schedule.MultiLevelTilingWithIntrin")
class MultiLevelTilingWithIntrin(ScheduleRule):
    """Extension of MultiLevelTiling for auto-tensorizing with a single intrinsic.

    Parameters
    ----------
    intrin_name : str
        The name of a tensor intrinsic, must be registerd via TensorIntrin.register(...) beforehand
    structure : str
        The tiling structure. Recommended:
        - 'SSRSRS' on CPU
        - 'SSSRRSRS' on GPU
    tile_bind : Optional[List[str]]
        For each level of tiles, which thread axis it is bound to. Recommended:
        - None on CPU
        - [blockIdx.x, vthread.x, threadIdx.x] on GPU
    max_innermost_factor : Optional[int]
        The maximum size of the innermost factor. None means no limit
    vector_load_lens : Optional[List[int]]
        The length of vector lane in vectorized cooperative fetching.
        None means disable vectorization
    reuse_read : Optional[ReuseType]
        Data reuse configuration for reading. None means no reuse.
    reuse_write : Optional[ReuseType]
        Data reuse configuration for writing. None means no reuse.
    """

    def __init__(
        self,
        intrin_name: str,
        structure: str,
        tile_binds: Optional[List[str]] = None,
        max_innermost_factor: Optional[int] = None,
        vector_load_lens: Optional[List[int]] = None,
        reuse_read: Optional[ReuseType] = None,
        reuse_write: Optional[ReuseType] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRuleMultiLevelTilingWithIntrin,  # type: ignore # pylint: disable=no-member
            intrin_name,
            structure,
            tile_binds,
            max_innermost_factor,
            vector_load_lens,
            reuse_read.as_dict() if reuse_read is not None else None,
            reuse_write.as_dict() if reuse_write is not None else None,
        )


@register_object("meta_schedule.MultiLevelTilingTensorCore")
class MultiLevelTilingTensorCore(ScheduleRule):
    """Extension of MultiLevelTiling for auto-tensorizing with multiple groups of candidate tensor
    core intrinsics.

    Parameters
    ----------
    intrin_groups : List[Mapping[str, str]]
        A list of groups of tensor core intrinsics. The map should contains key "init", "load_a",
        "load_b", "compute", "store", which represent the tensor intrin for initialization,
        loading operand A, loading operand B, tensor core computation, storing the result.
        The value of the map should be names of tensor intrinsics, must be registerd via
        TensorIntrin.register(...) beforehand
    structure : str
        The tiling structure. Recommended:
        - 'SSSRRSRS' on GPU
    tile_bind : Optional[List[str]]
        For each level of tiles, which thread axis it is bound to. Recommended:
        - [blockIdx.y, vthread.x, threadIdx.y] on GPU
    max_innermost_factor : Optional[int]
        The maximum size of the innermost factor. None means no limit
    vector_load_lens : Optional[List[int]]
        The length of vector lane in vectorized cooperative fetching.
        None means disable vectorization
    reuse_read : Optional[ReuseType]
        Data reuse configuration for reading. None means no reuse.
    reuse_write : Optional[ReuseType]
        Data reuse configuration for writing. None means no reuse.
    use_software_pipeline : bool
        Whether to use the software pipeline.
    """

    def __init__(
        self,
        intrin_groups: List[Mapping[str, str]],
        structure: str,
        tile_binds: Optional[List[str]] = None,
        max_innermost_factor: Optional[int] = None,
        vector_load_lens: Optional[List[int]] = None,
        reuse_read: Optional[ReuseType] = None,
        reuse_write: Optional[ReuseType] = None,
        use_software_pipeline: bool = False,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRuleMultiLevelTilingTensorCore,  # type: ignore # pylint: disable=no-member
            intrin_groups,
            structure,
            tile_binds,
            max_innermost_factor,
            vector_load_lens,
            reuse_read.as_dict() if reuse_read is not None else None,
            reuse_write.as_dict() if reuse_write is not None else None,
            use_software_pipeline,
        )


@register_object("meta_schedule.MultiLevelTilingWideVector")
class MultiLevelTilingWideVector(ScheduleRule):
    """Extension of MultiLevelTiling for backends with wide vectors. The loop over the innermost
    spatial axis of the output buffer is always vectorized with the maximum vector length.

    Parameters
    ----------
    structure : str
        The tiling structure. 'SSRSRS' is recommended.
    vector_length_in_bits: int
        The length of a vector register in bits.
    max_innermost_factor : Optional[int]
        The maximum size of the innermost factor. None means no limit
    reuse_read : Optional[ReuseType]
        Data reuse configuration for reading. None means no reuse.
    reuse_write : Optional[ReuseType]
        Data reuse configuration for writing. None means no reuse.
    """

    def __init__(
        self,
        structure: str,
        vector_length_in_bits: int,
        max_innermost_factor: Optional[int] = None,
        reuse_read: Optional[ReuseType] = None,
        reuse_write: Optional[ReuseType] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRuleMultiLevelTilingWideVector,  # type: ignore # pylint: disable=no-member
            structure,
            vector_length_in_bits,
            max_innermost_factor,
            reuse_read.as_dict() if reuse_read is not None else None,
            reuse_write.as_dict() if reuse_write is not None else None,
        )

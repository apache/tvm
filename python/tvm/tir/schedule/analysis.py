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
"""Analysis used in TensorIR scheduling"""
from typing import List, Optional

import tvm._ffi
from tvm.runtime import Object

from ..buffer import Buffer
from ..stmt import For
from ..expr import PrimExpr
from ..function import IndexMap, PrimFunc

from . import _ffi_api
from .schedule import Schedule, BlockRV


def suggest_index_map(
    buffer: Buffer,
    indices: List[PrimExpr],
    loops: List[For],
    predicate: PrimExpr,
) -> Optional[IndexMap]:
    """Provided the access pattern to a buffer, suggest one of the possible layout
    transformation to maximize the locality of the access pattern.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be transformed.
    indices : List[PrimExpr]
        The access pattern to the buffer.
    loops : List[For]
        The loops above the buffer.
    predicate : PrimExpr
        The predicate of the access.

    Returns
    -------
    index_map : Optional[IndexMap]
        The suggested index map. None if no transformation is suggested.
    """
    return _ffi_api.SuggestIndexMap(  # type: ignore # pylint: disable=no-member
        buffer,
        indices,
        loops,
        predicate,
    )


@tvm._ffi.register_object("tir.schedule.TensorizeInfo")
class TensorizeInfo(Object):
    """Necessary information used for tensorization."""


def get_tensorize_loop_mapping(
    sch: Schedule, block: BlockRV, desc_func: PrimFunc, allow_padding: bool = False
) -> Optional[TensorizeInfo]:
    """Establish a mapping between loops in a target block and an intrinsic description

    Parameters
    ----------
    sch : Schedule
        The schedule to be tensorized
    block : BlockRV
        The target block to match against
    desc_func : PrimFunc
        The prim func describing the computation to be tensorized
    allow_padding : bool
        Whether to allow padding the block iters to match the intrinsic description
    Returns
    -------
    tensorize_info : Optional[TensorizeInfo]
        TensorizeInfo structure if a valid mapping is found, None otherwise
    """
    return _ffi_api.GetTensorizeLoopMapping(sch, block, desc_func, allow_padding)  # type: ignore


@tvm._ffi.register_object("tir.schedule.AutoTensorizeMappingInfo")
class AutoTensorizeMappingInfo(Object):
    """Necessary information used to perform transformations for tensorization."""


def get_auto_tensorize_mapping_info(
    sch: Schedule, block: BlockRV, desc_func: PrimFunc
) -> Optional[AutoTensorizeMappingInfo]:
    """Get mapping info between a target block and an intrinsic description including layout
    transformations to apply.

    Parameters
    ----------
    sch : Schedule
        The schedule to be tensorized
    block : BlockRV
        The compute block for auto tensorization
    desc_func : PrimFunc
        The prim func describing the computation to be tensorized

    Returns
    -------
    auto_tensorize_mapping_info : Optional[AutoTensorizeMappingInfo]
        AutoTensorizeMappingInfo structure if potential mappings found, None otherwise.

    Note
    ----
    Returning a valid AutoTensorizeMappingInfo doesn't guarantee the block can be tensorized.
    We will need to apply the suggested layout transformations and then match against the tensor
    intrinsics.
    """
    return _ffi_api.GetAutoTensorizeMappingInfo(sch, block, desc_func)  # type: ignore


def has_block(sch: Schedule, block_name: str) -> bool:
    """Query if the given block name exists in the module associated with the provided schedule.

    Parameters
    ----------
    sch : Schedule
        The schedule
    block_name : str
        The name of the block to query

    Returns
    -------
    yes/no: bool
        True if the given block exists in the schedule.
    """
    return _ffi_api.HasBlock(sch, block_name)  # type: ignore


def is_output_block(sch: Schedule, block: BlockRV) -> bool:
    """Check whether the given block is an output block

    Parameters
    ----------
    sch : Schedule
        The schedule object of the block
    block : BlockRV
        The blockRV to be checked

    Returns
    -------
    yes/no : bool
        True if the given block is an output block

    """
    return _ffi_api.IsOutputBlock(sch, block)  # type: ignore

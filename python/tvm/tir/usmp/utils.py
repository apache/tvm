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
"""USMP Utilities and Data Structures"""
# pylint: disable=invalid-name

from typing import Optional, List

import tvm
from tvm._ffi import register_object
from tvm.runtime import Object
from . import _ffi_api
from ...ir.memory_pools import PoolInfo


# The allocate node attribute to indicate candidate memory pools.
# This needs to be kept in sync with CANDIDATE_MEMORY_POOL_ATTR in
# include/tvm/tir/usmp/utils.h
CANDIDATE_MEMORY_POOL_ATTR = "candidate_memory_pools"


def use_workspace_io_is_enabled() -> bool:
    """
    Check whether placing I/O tensors in the workspace is enabled.
    """
    ctx = tvm.transform.PassContext.current()
    return bool(ctx.config.get("tir.usmp.use_workspace_io", False))


@register_object("tir.usmp.BufferInfo")
class BufferInfo(Object):
    """BufferInfo object holds information related to buffers
    that are associated with tir.allocates and tir.allocate_consts
    that will be used with USMP

    Parameters
    ----------
    name_hint : str
        The name associated with the buffer (derived from TIR)

    size_bytes : int
        The size in bytes

    pool_candidates : List[PoolInfo]
        The list of candidates pools this buffer could be placed

    alignment : Optional[int]
        The byte alignment required in the workspace memory

    """

    def __init__(
        self,
        name_hint: str,
        size_bytes: int,
        pool_candidates: List[PoolInfo],
        alignment: Optional[int] = None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.BufferInfo,  # type: ignore # pylint: disable=no-member
            name_hint,
            size_bytes,
            pool_candidates,
            alignment,
        )

    def set_conflicts(self, conflicts: list):
        """Sets the conflicting array of buffer info objects"""
        _ffi_api.BufferInfoSetConflicts(self, conflicts)


@register_object("tir.usmp.PoolAllocation")
class PoolAllocation(Object):
    """PoolAllocation object holds information related to an allocation
    that indicates an offset in a pool

    Parameters
    ----------
    pool_info : PoolInfo
        The PoolInfo to which this allocation corresponds to

    byte_offset : int
        The offset in the pool where the allocate node should be placed

    """

    def __init__(self, pool_info: PoolInfo, byte_offset: int):
        self.__init_handle_by_constructor__(
            _ffi_api.PoolAllocation,  # type: ignore # pylint: disable=no-member
            pool_info,
            byte_offset,
        )

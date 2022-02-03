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

from typing import Dict, Optional, List, Union

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.target import Target
from . import _ffi_api


# The allocate node attribute to indicate candidate memory pools.
# This needs to be kept in sync with CANDIDATE_MEMORY_POOL_ATTR in
# include/tvm/tir/usmp/utils.h
CANDIDATE_MEMORY_POOL_ATTR = "candidate_memory_pools"


@register_object("tir.usmp.PoolInfo")
class PoolInfo(Object):
    """PoolInfo object holds information related to memory pools
    where the statically sized allocate nodes will pooled into.

    Parameters
    ----------
    pool_name : str
        The name of the memory pool

    target_access : Dict[Target, str]
        A dictionary where keys describe which targets could
        access the pool where value could take the values :
        a) "rw" : read-write access
        b) "ro" : write-only acesss

    size_hint_bytes : Optional[int]
        The expected size hint to be used by the allocator.
        The default value would be -1 which means the pool
        is not size restricted.

    clock_frequency_hz : Optional[int]
        The clock frequency that the memory pool runs at in Hz.
        If not specified/known, this will default to -1 indicating
        it hasn't been defined.

    read_bandwidth_bytes_per_cycle : Optional[int]
        The read bandwidth of the memory pool in bytes/cycle.
        If not specified/known, this will default to -1 indicating
        it hasn't been defined.

    write_bandwidth_bytes_per_cycle : Optional[int]
        The write bandwidth of the memory pool in bytes/cycle.
        If not specified/known, this will default to -1 indicating
        it hasn't been defined.

    read_latency_cycles : Optional[int]
        The read latency of the memory pool in cycles.
        If not specified/known, this will default to 0.

    write_latency_cycles : Optional[int]
        The write latency of the memory pool in cycles.
        If not specified/known, this will default to 0.

    target_burst_bytes : Optional[Union[Dict[Target, int], None]]
        The burst length of the memory pool in bytes per target.
        If not specified/known for a given target, a burst length
        of 1 byte will be assumed.

    """

    # The string parameter to indicate read and write access to a pool
    # This needs to be kept in sync with kTargetPoolReadWriteAccess in
    # include/tvm/tir/usmp/utils.h
    READ_WRITE_ACCESS = "rw"
    # The string parameter to indicate read only access to a pool
    # This needs to be kept in sync with kTargetPoolReadOnlyAccess in
    # include/tvm/tir/usmp/utils.h
    READ_ONLY_ACCESS = "ro"

    def __init__(
        self,
        pool_name: str,
        target_access: Dict[Target, str],
        size_hint_bytes: Optional[int] = -1,
        clock_frequency_hz: Optional[int] = -1,
        read_bandwidth_bytes_per_cycle: Optional[int] = -1,
        write_bandwidth_bytes_per_cycle: Optional[int] = -1,
        read_latency_cycles: Optional[int] = 0,
        write_latency_cycles: Optional[int] = 0,
        target_burst_bytes: Optional[Union[Dict[Target, int], None]] = None,
    ):
        if not target_burst_bytes:
            target_burst_bytes = dict()

        self.__init_handle_by_constructor__(
            _ffi_api.PoolInfo,  # type: ignore # pylint: disable=no-member
            pool_name,
            target_access,
            size_hint_bytes,
            clock_frequency_hz,
            read_bandwidth_bytes_per_cycle,
            write_bandwidth_bytes_per_cycle,
            read_latency_cycles,
            write_latency_cycles,
            target_burst_bytes,
        )


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
        """Sets the the conflicting array of buffer info objects"""
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

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
"""Objects for Memory Pools to be used within the compilation"""

from typing import Optional, List

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.runtime import NDArray
from . import _ffi_api


@register_object("ir.PoolInfo")
class PoolInfo(Object):
    """PoolInfo object holds information related to memory pools
    where the statically sized allocate nodes will pooled into.
    This is a base class for WorkspacePoolInfo and ConstantPoolInfo.
    """

    def __init__(self):
        pass


@register_object("ir.PoolInfoProperties")
class PoolInfoProperties(Object):
    """PoolInfo object holds information related to memory pools
    where the statically sized allocate nodes will pooled into.

    Parameters
    ----------
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

    def __init__(
        self,
        size_hint_bytes: Optional[int] = -1,
        clock_frequency_hz: Optional[int] = -1,
        read_bandwidth_bytes_per_cycle: Optional[int] = -1,
        write_bandwidth_bytes_per_cycle: Optional[int] = -1,
        read_latency_cycles: Optional[int] = 0,
        write_latency_cycles: Optional[int] = 0,
        target_burst_bytes=None,
    ):
        if not target_burst_bytes:
            target_burst_bytes = dict()

        self.__init_handle_by_constructor__(
            _ffi_api.PoolInfoProperties,  # type: ignore # pylint: disable=no-member
            size_hint_bytes,
            clock_frequency_hz,
            read_bandwidth_bytes_per_cycle,
            write_bandwidth_bytes_per_cycle,
            read_latency_cycles,
            write_latency_cycles,
            target_burst_bytes,
        )


@register_object("ir.ConstantInfo")
class ConstantInfo(Object):
    """ConstantInfo object hold information on a constant pool.

    Parameters
    ----------
    name_hint : str
        Name of the constant.
    byte_offset : int
        The byte_offset of the constant.
    data : NDArray
        The data of the constant.
    """

    def __init__(
        self,
        name_hint: str,
        byte_offset: int,
        data: NDArray,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.ConstantInfo,  # type: ignore # pylint: disable=no-member
            name_hint,
            byte_offset,
            data,
        )


@register_object("ir.WorkspacePoolInfo")
class WorkspacePoolInfo(PoolInfo):
    """WorkspacePoolInfo object holds information related to RW memory pools
    where the statically sized allocate nodes will pooled into.

    Parameters
    ----------
    pool_name : str
        The name of the memory pool

    targets : list[Target]
        A list of targets which could access the pool

    pool_info_properties : PoolInfoProperties
        The properties of the pool.
    """

    def __init__(
        self,
        pool_name: str,
        targets,
        pool_info_properties=None,
    ):
        super().__init__()

        if pool_info_properties is None:
            pool_info_properties = PoolInfoProperties()

        self.__init_handle_by_constructor__(
            _ffi_api.WorkspacePoolInfo,  # type: ignore # pylint: disable=no-member
            pool_name,
            targets,
            pool_info_properties,
        )


@register_object("ir.ConstantPoolInfo")
class ConstantPoolInfo(PoolInfo):
    """ConstantPoolInfo object holds information related to RO memory pools
    where the statically sized allocate nodes are pooled into.

    Parameters
    ----------
    pool_name : str
        The name of the memory pool

    targets : list[Target]
        describes which targets could access the pool

    pool_info_properties : PoolInfoProperties
        The properties of the pool.
    """

    def __init__(
        self,
        pool_name: str,
        targets,  # list[Target]
        constant_info_arr=None,  # list[ConstantInfo]
        pool_info_properties=None,
    ):
        super().__init__()

        if constant_info_arr is None:
            constant_info_arr = []
        if pool_info_properties is None:
            pool_info_properties = PoolInfoProperties()
        self.__init_handle_by_constructor__(
            _ffi_api.ConstantPoolInfo,  # type: ignore # pylint: disable=no-member
            pool_name,
            targets,
            constant_info_arr,
            pool_info_properties,
        )


@register_object("ir.WorkspaceMemoryPools")
class WorkspaceMemoryPools(Object):
    """This object contains a list of WorkspacePoolInfo objects to be used as
    workspace memory in the compilation

    Parameters
    ----------
    pools : List[WorkspacePoolInfo]
        The list of ConstantPoolInfo objects to be used with the compilation
    """

    def __init__(
        self,
        pools: List[WorkspacePoolInfo],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.WorkspaceMemoryPools, pools  # type: ignore # pylint: disable=no-member
        )


@register_object("ir.ConstantMemoryPools")
class ConstantMemoryPools(Object):
    """This object contains a list of ConstantPoolInfo objects to be used as
    read-only memory in the compilation

    Parameters
    ----------
    pools : List[ConstantPoolInfo]
        The list of ConstantPoolInfo objects to be used with the compilation
    """

    def __init__(
        self,
        pools: List[ConstantPoolInfo],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.ConstantMemoryPools, pools  # type: ignore # pylint: disable=no-member
        )


@register_object("ir.AllocatedPoolInfo")
class AllocatedPoolInfo(Object):
    """Allocate memory in a given pool.

    Parameters
    ----------
    pool : PoolInfo
        The pool in which to allocate memory.
    allocated_size : int
        The size of memory to allocate.
    """

    def __init__(
        self,
        pool: PoolInfo,
        allocated_size: int,
        pool_var_idx: int = 0,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.AllocatedPoolInfo, pool, allocated_size, pool_var_idx  # type: ignore # pylint: disable=no-member
        )

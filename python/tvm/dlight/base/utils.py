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
# pylint: disable=missing-docstring
"""Utility methods for generic GPU."""
from typing import List, Optional, Union

from tvm import DataType, tir
from tvm.target import Target


def get_bytes(dtype: Union[DataType, str]) -> int:
    if isinstance(dtype, str):
        dtype = DataType(dtype)
    return dtype.itemsize()


def get_extent(sch: tir.Schedule, loop_rv: tir.schedule.LoopRV):
    loop: tir.For = sch.get(loop_rv)
    return loop.extent.value if isinstance(loop.extent, tir.IntImm) else loop.extent


def auto_vectorize(sch: tir.Schedule, loop: tir.schedule.LoopRV, max_vec: int):
    """Auto vectorize the loop."""
    extent = get_extent(sch, loop)
    if not isinstance(extent, int):
        return
    v = loop if extent <= max_vec else sch.split(loop, factors=[None, max_vec])[-1]
    sch.vectorize(v)


def max_threads_per_block(target: Target) -> int:
    """Get the maximum number of threads per block for a given target.

    Parameters
    ----------
    target : Target
        The target to get the maximum number of threads per block for.

    Returns
    -------
    max_threads_per_block : int
        The maximum number of threads per block for the given target.
    """
    for name in ["max_threads_per_block", "max_num_threads"]:
        result = target.attrs.get(name, None)
        if result is not None:
            return result
    if target.kind.name == "cuda":
        return 1024
    return 256


def suggest_threads_per_block(
    target: Target,
    loops: List[tir.For],
    max_threads_for_dynamic_loop: int = 32,
) -> List[int]:
    if target.kind.name == "cuda":
        threads = 1024
    elif target.kind.name == "rocm":
        threads = 256
    elif target.kind.name == "metal":
        threads = 256
    elif target.kind.name == "opencl":
        threads = 256
    else:
        threads = 64
    results: List[Optional[int]] = []
    dynamic: List[int] = []
    for i, loop in enumerate(loops):
        loop_extent = loop.extent
        if isinstance(loop_extent, tir.IntImm):
            loop_extent = loop_extent.value
            extent = 1
            while extent <= loop_extent and extent <= threads:
                extent *= 2
            extent //= 2
            assert extent >= 1
            assert threads % extent == 0
            threads //= extent
            results.append(extent)
        else:
            results.append(None)
            dynamic.append(i)

    for i in dynamic:
        extent = 1
        while extent <= max_threads_for_dynamic_loop and extent <= threads:
            extent *= 2
        extent //= 2
        assert extent >= 1
        assert threads % extent == 0
        threads //= extent
        results[i] = extent

    if dynamic:
        results[dynamic[0]] *= threads

    return results

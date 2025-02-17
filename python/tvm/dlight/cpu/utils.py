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
"""Utility methods for generic CPU."""
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


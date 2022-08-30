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
"""USMP Transform Python API for passes"""
# pylint: disable=invalid-name

from typing import Dict

import tvm
from tvm.tir import Stmt
from tvm.tir.usmp.utils import PoolAllocation
from . import _ffi_api


def convert_pool_allocations_to_offsets(
    pool_allocations: Dict[Stmt, PoolAllocation], emit_tvmscript_printable: bool = False
) -> tvm.transform.Pass:
    """Convert pool allocations to Load nodes with offsets from pools.

    Parameters
    ----------
    pool_allocations : Dict[Stmt, PoolAllocation]
        Allocate or AllocateConst node to pool allocation mapping
    emit_tvmscript_printable : bool
        A toggle to emit TVMScript printable IRModule for unit tests
        removing all attributes that should be attached for integration

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that converts the allocations to offsets.
    """
    return _ffi_api.ConvertPoolAllocationsToOffsets(pool_allocations, emit_tvmscript_printable)

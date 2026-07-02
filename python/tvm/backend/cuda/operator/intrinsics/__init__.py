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
# pylint: disable=unused-import
"""CUDA HW intrinsic codegens, grouped by feature domain.

- ``mma`` / ``wgmma`` / ``tcgen05`` — matrix-multiply hardware (Volta+/Hopper/Blackwell).
- ``cp_async`` — cp.async + cp.async.bulk + cp.async.bulk.tensor (TMA), incl. TMA address helpers.
- ``sync`` — barriers, fences, mbarrier, cluster.barrier, warp vote, elect, sync helpers.
- ``math`` — packed-f32x2 arithmetic, exp2/rcp/reduce3, warp/CTA reductions.
- ``memory`` — typed copies, ldg, ld.global.acquire, atomics, type conversions, address casts.
- ``nvshmem`` — NVSHMEM RMA / signal / collective.
- ``misc`` — register-allocation control, profiler timer, debug helpers (printf / trap).

Plus the support modules:

- ``header`` — CUDA header generator and helper-tag table.
- ``registry`` — codegen registry and manifest grouped by CUDA feature namespace.
- ``types`` — PTX dtype enum.
- ``utils`` — small parsing / validation helpers.
"""

# Import op modules to register their codegen functions.
from . import cp_async, math, memory, misc, mma, nvshmem, sync, tcgen05, wgmma
from .header import TAGS, header_generator
from .registry import (
    CODEGEN_MANIFEST,
    CODEGEN_REGISTRY,
    get_codegen,
    list_registered_codegen,
    register_codegen,
)
from .types import PTXDataType

__all__ = [
    "CODEGEN_REGISTRY",
    "CODEGEN_MANIFEST",
    "TAGS",
    "PTXDataType",
    "get_codegen",
    "header_generator",
    "list_registered_codegen",
    "register_codegen",
]

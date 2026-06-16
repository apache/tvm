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
"""Shared enum / value tables for PTX intrinsic schemas and user wrappers.

Single source of truth. Both ``tvm.tirx.op`` (user wrappers that validate
arguments via ``_choice``) and ``tvm.tirx.cuda.operator.intrinsics.*``
(schema declarations using ``Choice(choices=...)`` / ``IntAttr(choices=...)``)
import from here.

Adding a new modifier value requires changing exactly one place.
"""

# Memory ordering / scope -----------------------------------------------------
FENCE_SEM = ("sc", "acq_rel")
FENCE_SCOPE = ("cta", "cluster", "gpu", "sys")
FENCE_PROXY_ASYNC_SPACE = ("", "global", "shared::cta", "shared::cluster")
CLUSTER_BARRIER_SEM = ("", "release", "relaxed")

# CTA group (used by tcgen05 and TMA) -----------------------------------------
TCGEN05_CTA_GROUP = (1, 2)

# NVSHMEM ---------------------------------------------------------------------
NVSHMEM_CMP = ("eq", "ne", "gt", "ge", "lt", "le")
NVSHMEM_SIG_OP = ("set", "add")

# Floating-point rounding -----------------------------------------------------
F32X2_ROUND = ("rz", "rn", "rm", "rp")

# cp.async (non-bulk) ---------------------------------------------------------
CP_ASYNC_CACHE_HINT = ("", "evict_last", "evict_first", "evict_normal")
CP_ASYNC_PREFETCH_SIZE = (-1, 64, 128, 256)
CP_ASYNC_FILL_MODE = ("", "zero")

# cp.async.bulk (TMA) ---------------------------------------------------------
CP_ASYNC_BULK_CACHE_HINT = ("", "evict_last", "evict_first", "evict_normal", "evict_last_use")
CP_ASYNC_BULK_RED_OP = ("add", "min", "max", "inc", "dec", "and", "or", "xor")

# ldmatrix / stmatrix ---------------------------------------------------------
LDMATRIX_DTYPE = (".b16", ".b8")
LDMATRIX_NUM = (1, 2, 4)

# tcgen05.cp ------------------------------------------------------------------
TCGEN05_CP_SHAPES = ("32x128b", "4x256b", "128x128b", "128x256b", "64x128b")
TCGEN05_CP_MULTICAST = ("", "warpx4", "warpx2::02_13", "warpx2::01_23")
TCGEN05_CP_DECOMPRESS = ("", "b8x16.b4x16_p64", "b8x16.b6x16_p32")

# tcgen05.ld / tcgen05.st -----------------------------------------------------
TCGEN05_LDST_SHAPES = ("16x32bx2", "16x64b", "16x128b", "16x256b", "32x32b")

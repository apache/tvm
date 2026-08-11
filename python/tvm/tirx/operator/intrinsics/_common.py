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
"""Shared enum / value tables for the PTX intrinsic user wrappers.

Single source of truth for the modifier values ``tvm.tirx.op`` accepts, which
it validates via ``_choice``. The ``T.ptx`` dialect does not read these -- it
derives its legal modifier tokens from its own instruction table.

Adding a new modifier value requires changing exactly one place.
"""

# Memory ordering / scope -----------------------------------------------------
CLUSTER_BARRIER_SEM = ("", "release", "relaxed")
MBARRIER_COMPLETE_TX_SEM = ("relaxed",)
MBARRIER_COMPLETE_TX_SCOPE = ("cta", "cluster")
MBARRIER_COMPLETE_TX_SPACE = ("shared", "shared::cta", "shared::cluster")
MBARRIER_ARRIVE_SEM = ("", "release", "relaxed")
MBARRIER_ARRIVE_SCOPE = ("", "cta", "cluster")
MBARRIER_ARRIVE_SPACE = ("shared", "shared::cta", "shared::cluster")

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

# tcgen05.ld / tcgen05.st -----------------------------------------------------
TCGEN05_LDST_SHAPES = ("16x32bx2", "16x64b", "16x128b", "16x256b", "32x32b")

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

copy_async
==========

Asynchronous copy. Every variant emits the transfer's *issue* instruction; the
caller supplies the matching completion protocol. ``ldgsts`` uses ``cp.async``
commit/wait, TMA loads and distributed-shared-memory copies signal an mbarrier,
TMA stores use the bulk async-group commit/wait operations, and tensor-memory
paths use their ``tcgen05`` commit/wait operations. Selection is by the
source/destination memory pair and scope. CUDA currently registers six variant
names; the TMA page below covers two of them.

.. list-table::
   :header-rows: 1
   :widths: 26 16 14 44

   * - Variant
     - Pair
     - Prio
     - Issue instruction
   * - :doc:`copy_async/ldgsts`
     - global → shared
     - 20
     - ``cp.async`` (LDGSTS), per-thread vectorized
   * - :doc:`copy_async/tma` (``tma_auto`` / ``tma_explicit``)
     - global ↔ shared
     - 10
     - ``cp.async.bulk.tensor`` (TMA, descriptor-driven, single-thread)
   * - :doc:`copy_async/dsmem`
     - shared → shared (cross-CTA)
     - 10
     - ``cp.async.bulk`` shared::cluster (``mapa`` remote address)
   * - :doc:`copy_async/tcgen05_cp` (``smem->tmem``)
     - shared → tmem
     - 10
     - one of the registered ``tcgen05.cp`` shapes (matrix-descriptor driven)
   * - :doc:`copy_async/tcgen05_ldst` (``tmem<->local``)
     - tmem ↔ register
     - 10
     - ``tcgen05.ld`` / ``tcgen05.st`` (warpgroup, atom-matched)

.. toctree::
   :maxdepth: 1

   copy_async/ldgsts
   copy_async/tma
   copy_async/dsmem
   copy_async/tcgen05_cp
   copy_async/tcgen05_ldst

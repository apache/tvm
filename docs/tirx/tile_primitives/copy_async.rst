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

Asynchronous copy. Every variant emits only the *issue* instruction — the caller is
responsible for completion (``cp.async`` commit/wait for ``ldgsts``; mbarrier
arrive/wait for the bulk-tensor and dsmem paths; ``tcgen05.commit`` /
``tcgen05.wait`` for the tensor-memory paths). Selection is by the source/dest
memory pair and scope.

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
   * - :doc:`copy_async/tma`
     - global ↔ shared
     - 10
     - ``cp.async.bulk.tensor`` (TMA, descriptor-driven, single-thread)
   * - :doc:`copy_async/dsmem`
     - shared → shared (cross-CTA)
     - 10
     - ``cp.async.bulk`` shared::cluster (``mapa`` remote address)
   * - :doc:`copy_async/tcgen05_cp`
     - shared → tmem
     - 10
     - ``tcgen05.cp.32x128b.warpx4`` (matrix-descriptor driven)
   * - :doc:`copy_async/tcgen05_ldst`
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

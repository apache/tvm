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

copy
====

Synchronous element copy ``src → dst`` between global, shared, and register
(``local``) memory.  CUDA currently registers eight variants: five explicit
fixed-width variants, ``ldstmatrix``, ``vec_auto``, and ``fallback``.  The
``vec_auto`` implementation contains separate global/shared and register paths;
``gmem_smem`` and ``reg`` below name those implementation paths, not selectable
dispatch variants.

.. list-table::
   :header-rows: 1
   :widths: 28 16 12 44

   * - Variant
     - Pair
     - Prio
     - Lowering
   * - ``vec_16b`` / ``vec_32b`` / ``vec_64b`` / ``vec_128b`` /
       ``vec_256b``
     - any supported pair
     - 20
     - explicit thread-scope transfer of exactly the named width; optional
       global-load cache controls
   * - ``vec_auto``: :doc:`copy/gmem_smem` path
     - global ↔ shared
     - 10
     - synthesized ``[outer, threads, vec]`` partition, vectorized ``copy_Nb``
   * - ``vec_auto``: :doc:`copy/reg` path
     - register ↔ shared/global
     - 10
     - partition induced by the register layout's thread axes
   * - :doc:`copy/ldstmatrix`
     - register ↔ shared
     - 10
     - warp-collective ``ldmatrix`` / ``stmatrix`` (m8n8 fragments)
   * - :doc:`copy/fallback`
     - any
     - 0
     - scalar single-thread copy (catch-all)

The detailed pages cover the two ``vec_auto`` paths, ``ldstmatrix``, and the
fallback, including accepted input, lowering algorithm, emitted TIRx IR, and
generated CUDA:

.. toctree::
   :maxdepth: 1

   copy/gmem_smem
   copy/reg
   copy/ldstmatrix
   copy/fallback

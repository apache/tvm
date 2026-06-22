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
(``local``) memory. Four CUDA variants are registered. ``gmem_smem`` (global ↔
shared) and the two register variants take disjoint memory-scope pairs; ``reg`` and
``ldstmatrix`` both cover ``register ↔ shared``, where ``ldstmatrix`` is tried first
and **declines to** ``reg`` unless the layouts are ldmatrix fragments. All three are
priority 10; the scalar ``fallback`` (priority 0) runs only if they all decline.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Variant
     - Pair
     - Lowering
   * - :doc:`copy/gmem_smem`
     - global ↔ shared
     - synthesized ``[outer, threads, vec]`` partition, vectorized ``copy_Nb``
   * - :doc:`copy/reg`
     - register ↔ shared/global
     - partition induced by the register layout's thread axes
   * - :doc:`copy/ldstmatrix`
     - register ↔ shared
     - warp-collective ``ldmatrix`` / ``stmatrix`` (m8n8 fragments)
   * - :doc:`copy/fallback`
     - any
     - scalar single-thread copy (priority 0, catch-all)

Each variant has its own deep walkthrough — accepted input, algorithm, a runnable
demo program, the dispatch's TIRx IR, and the generated CUDA:

.. toctree::
   :maxdepth: 1

   copy/gmem_smem
   copy/reg
   copy/ldstmatrix
   copy/fallback

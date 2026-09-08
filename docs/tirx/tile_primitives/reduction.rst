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

reduction
=========

Covers ``sum``, ``max``, ``min`` (reduce over ``axes``).  All three operations
register ``local`` and ``shared`` variants at priority 10, discriminated by
operand storage scope.  On CUDA SM100 or newer, ``sum`` additionally registers
``packed_add_sum`` and ``max`` / ``min`` register ``3input_maxmin`` at priority
20.  The implementation file and detailed page group those two names as the
SM100 packed paths; ``sm100_packed`` is not itself a dispatch name.

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Variant
     - Prio
     - Lowering
   * - :doc:`reduction/local`
     - 10
     - local-buffer src/dst; sequential thread reduction (+ optional warp shuffle)
   * - :doc:`reduction/shared`
     - 10
     - shared src/dst; adaptive group-size ``__shfl_xor`` tree
   * - :doc:`reduction/sm100_packed` (``packed_add_sum`` / ``3input_maxmin``)
     - 20
     - CUDA SM100+ thread-scope fp32 ≥8: packed ``add.f32x2`` /
       ``max3``/``min3``

.. toctree::
   :maxdepth: 1

   reduction/local
   reduction/shared
   reduction/sm100_packed

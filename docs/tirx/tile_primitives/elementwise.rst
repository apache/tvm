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

elementwise
===========

Covers ``cast``, ``fill``, the unary ops (``zero``, ``reciprocal``, ``sqrt``,
``exp``, ``exp2``, ``log2``, ``silu``), the binary ops (``add``, ``sub``,
``mul``, ``fdiv``, ``maximum``), and ``fma``. Every op registers **two** variants — ``reg`` and
``smem`` — both at priority 10; the buffer-operand storage scope (all-local vs
all-shared) is the mutually-exclusive discriminator. Scalar inputs do not have a
storage scope. Each op is described by an
``OpSpec`` (a ``parse`` that builds the destination + source list, optional dtype
checks, and the scalar expression applied per element).

This is the current CUDA elementwise set. Other constructors in the global
``Tx.tile`` catalog, including ``minimum``, ``memset``, and ``select``, do not
currently have these CUDA ``reg`` / ``smem`` variants; a call needs a variant
registered by its selected target backend.

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Variant
     - Operands
     - Lowering
   * - :doc:`elementwise/reg`
     - all buffer operands local
     - partition induced by the local-buffer layout; op applied per-thread
   * - :doc:`elementwise/smem`
     - all buffer operands shared
     - synthesized ``[outer, threads, vec]`` partition; op applied per (vectorized)
       element

.. toctree::
   :maxdepth: 1

   elementwise/reg
   elementwise/smem

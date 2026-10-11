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

CUDA Tensor Loads, Stores, and Moves
====================================

``T.cuda.tile.ld(dst_registers, src_memory, scope=...)`` and
``T.cuda.tile.st(dst_memory, src_registers, scope=...)`` expand the register
layout into memory instructions. ``vec_bits`` requests a fixed-width
thread-level instruction. ``ldmatrix`` and ``stmatrix`` explicitly select
matrix load/store instructions. ``mov`` copies registers or fills a tensor.

Synchronous global/shared copying requires explicit local storage and separate
load and store operations, or caller-written scalar loops. There is no scalar
fallback dispatcher. See :doc:`../tile_primitives` for scope and cache
qualifiers.

.. toctree::
   :maxdepth: 1

   copy/reg
   copy/ldstmatrix

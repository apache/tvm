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

TIRx Basics: CUDA C++/PTX native level
======================================

.. note::

   Native-level kernel authoring for the **CUDA backend** (the ``"cuda"``
   target): the thread hierarchy, memory scopes, the ``T.cuda.*`` / ``T.ptx.*``
   intrinsics, and the compile / run / inspect loop. The complete kernels in
   these chapters (``scale``, ``add``, ``smem_demo``, ``block_sum``, and the
   warp all-reduce) are tested end-to-end on a CUDA GPU.

What "native level" means
-------------------------

A native-level TIRx kernel reads like a structured device kernel: you place
threads yourself, allocate shared/register buffers, write loops and barriers, and
call device intrinsics directly. There is no automatic scheduling — what you write
is what is emitted. This is the foundation the tile primitives
(:doc:`tile_primitives`) are built on; everything here is what those primitives
ultimately lower to, so it is also where you go when a hardware feature does not
have a primitive yet.

The authoring model
-------------------

- ``@T.prim_func`` (or ``@T.jit`` for compile-time-specialized) kernels, written
  with ``from tvm.script import tirx as T``;
- ``T.device_entry()`` plus *scope-id* intrinsics for thread binding;
- ``T.match_buffer`` parameters and ``T.alloc_*`` scratch buffers;
- ordinary loops, branches, and scalar math;
- ``tvm.compile(mod, target=..., tir_pipeline="tirx")`` to build, then call the
  result directly.

All native authoring uses these imports. The ``__future__`` import lets ``@T.jit``
kernels reference compile-time parameters inside type annotations (see
:doc:`native_basics/cuda/functions`); it is harmless for ordinary kernels::

    from __future__ import annotations
    import tvm
    from tvm.script import tirx as T

.. toctree::
   :maxdepth: 1

   native_basics/cuda/first_kernel
   native_basics/cuda/functions
   native_basics/cuda/parser_utils
   native_basics/cuda/data_types
   native_basics/cuda/buffers
   native_basics/cuda/control_flow
   native_basics/cuda/threads_sync
   native_basics/cuda/compiling
   native_basics/cuda/profiling

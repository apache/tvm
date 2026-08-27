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

TIRx TVMScript
==============

The TIRx dialect is exposed as ``tvm.script.tirx``.  Kernel authors normally
use two aliases: ``T`` for lower-level IR construction and backend intrinsics,
and ``Tx`` for the higher-level tile-primitive facade::

   from tvm.script import tirx as T
   from tvm.script.tirx import tile as Tx

These are modules, not fixed language keywords.  Code that prefers a single
``Tx`` alias can spell the same three API layers as follows::

   from tvm.script import tirx as Tx

   # Core TIRx: loops, buffers, scalar expressions, and statements.
   Tx.alloc_buffer(...)

   # High-level tile primitive; equivalent to Tx.copy(...) with split aliases.
   Tx.tile.copy(...)

   # Direct CUDA instruction dialect; equivalent to T.ptx.* with split aliases.
   Tx.ptx.mbarrier.init.shared.b64(...)

.. list-table:: TIRx script layers
   :header-rows: 1
   :widths: 24 25 25 26

   * - Layer
     - Split aliases
     - Single ``Tx`` alias
     - Python implementation
   * - Core TIRx IR
     - ``T.*``
     - ``Tx.*``
     - ``tvm.tirx.script.builder.ir``
   * - Tile primitives
     - ``Tx.*``
     - ``Tx.tile.*``
     - ``tvm.tirx.script.tile``
   * - Direct CUDA PTX
     - ``T.ptx.*``
     - ``Tx.ptx.*``
     - ``tvm.backend.cuda.ptx``

``tvm.script.tirx`` is a syntax and construction frontend for the
``tvm.tirx`` object model, not a separate IR.  Core builders create nodes such
as ``tvm.tirx.For``, ``tvm.tirx.BufferLoad``, and ``tvm.ir.Call`` directly.

The high-level tile facade validates tile-only arguments before delegating to
the raw constructors.  Both entry points create the same
``tvm.tirx.TilePrimitiveCall`` node; the raw constructors are not a second,
lower IR.  During ``tvm.tirx.transform.LowerTIRx``, tile-primitive dispatch
selects a target implementation and replaces each ``TilePrimitiveCall`` with
ordinary TIRx statements and calls.  The direct PTX namespace creates
``tirx.ptx.*`` calls immediately and bypasses tile dispatch; see :doc:`ptx`.

Parser entry points
-------------------
.. automodule:: tvm.tirx.script.parser.entry
   :members:
   :no-index:
   :exclude-members: TIRInline

Core IR builder
---------------
.. automodule:: tvm.tirx.script.builder.ir
   :members:
   :no-index:
   :exclude-members: anylist_getitem, anylist_resetitem, anylist_setitem_call_packed, anylist_setitem_call_cpacked

High-level tile primitives
--------------------------
.. automodule:: tvm.tirx.script.tile
   :members: compose_op
   :no-index:

The scope namespaces bind a tile operation to an execution scope.  For example,
``Tx.copy(...)`` uses thread scope, while ``Tx.warp.copy(...)``,
``Tx.wg.copy(...)``, and ``Tx.cta.copy(...)`` request wider cooperation.

.. autodata:: tvm.tirx.script.tile.thread
   :no-index:

.. autodata:: tvm.tirx.script.tile.warp
   :no-index:

.. autodata:: tvm.tirx.script.tile.wg
   :no-index:

.. autodata:: tvm.tirx.script.tile.warpgroup
   :no-index:

.. autodata:: tvm.tirx.script.tile.cta
   :no-index:

.. autodata:: tvm.tirx.script.tile.cluster
   :no-index:

``ScopedOp`` objects are callable but are not ordinary Python functions, so
they are listed explicitly for autodoc.

.. autofunction:: tvm.tirx.script.tile.add
   :no-index:

.. autofunction:: tvm.tirx.script.tile.binary_chain
   :no-index:

.. autofunction:: tvm.tirx.script.tile.binary_reduce
   :no-index:

.. autofunction:: tvm.tirx.script.tile.cast
   :no-index:

.. autofunction:: tvm.tirx.script.tile.copy
   :no-index:

.. autofunction:: tvm.tirx.script.tile.copy_async
   :no-index:

.. autofunction:: tvm.tirx.script.tile.exp
   :no-index:

.. autofunction:: tvm.tirx.script.tile.exp2
   :no-index:

.. autofunction:: tvm.tirx.script.tile.log2
   :no-index:

.. autofunction:: tvm.tirx.script.tile.fdiv
   :no-index:

.. autofunction:: tvm.tirx.script.tile.fill
   :no-index:

.. autofunction:: tvm.tirx.script.tile.fma
   :no-index:

.. autofunction:: tvm.tirx.script.tile.gemm
   :no-index:

.. autofunction:: tvm.tirx.script.tile.gemm_async
   :no-index:

.. autofunction:: tvm.tirx.script.tile.max
   :no-index:

.. autofunction:: tvm.tirx.script.tile.maximum
   :no-index:

.. autofunction:: tvm.tirx.script.tile.memset
   :no-index:

.. autofunction:: tvm.tirx.script.tile.min
   :no-index:

.. autofunction:: tvm.tirx.script.tile.minimum
   :no-index:

.. autofunction:: tvm.tirx.script.tile.mul
   :no-index:

.. autofunction:: tvm.tirx.script.tile.permute_layout
   :no-index:

.. autofunction:: tvm.tirx.script.tile.reciprocal
   :no-index:

.. autofunction:: tvm.tirx.script.tile.reduce_negate
   :no-index:

.. autofunction:: tvm.tirx.script.tile.select
   :no-index:

.. autofunction:: tvm.tirx.script.tile.silu
   :no-index:

.. autofunction:: tvm.tirx.script.tile.sqrt
   :no-index:

.. autofunction:: tvm.tirx.script.tile.sub
   :no-index:

.. autofunction:: tvm.tirx.script.tile.sum
   :no-index:

.. autofunction:: tvm.tirx.script.tile.unary_reduce
   :no-index:

.. autofunction:: tvm.tirx.script.tile.zero
   :no-index:

Raw tile-primitive node constructors
------------------------------------
``tvm.tirx.script.builder.tirx`` constructs the same
``TilePrimitiveCall`` nodes as the validated facade above.  This construction
surface is useful to parser, builder, and extension authors; kernel code should
normally use ``tvm.tirx.script.tile``.

.. automodule:: tvm.tirx.script.builder.tirx
   :members:
   :no-index:
   :exclude-members: ScopeNamespace, ScopedOp

As above, the callable ``ScopedOp`` objects need explicit autodoc directives.

.. autofunction:: tvm.tirx.script.builder.tirx.add
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.binary_chain
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.binary_reduce
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.cast
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.copy
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.copy_async
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.exp
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.exp2
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.log2
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.fdiv
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.fill
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.fma
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.gemm
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.gemm_async
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.max
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.maximum
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.memset
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.min
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.minimum
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.mul
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.permute_layout
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.reciprocal
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.reduce_negate
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.select
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.silu
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.sqrt
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.sub
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.sum
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.unary_reduce
   :no-index:

.. autofunction:: tvm.tirx.script.builder.tirx.zero
   :no-index:

Dispatch extension points
-------------------------
.. automodule:: tvm.tirx.operator.tile_primitive
   :members:
   :imported-members:
   :no-index:

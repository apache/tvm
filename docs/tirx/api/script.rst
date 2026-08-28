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

The TIRx dialect is exposed as ``tvm.script.tirx``.  The documentation uses
``Tx`` as its ordinary Python module alias::

   from tvm.script import tirx as Tx

   # Core TIRx: loops, buffers, scalar expressions, and statements.
   Tx.alloc_buffer(...)

   # High-level tile primitive.
   Tx.tile.copy(...)

   # Direct CUDA instruction dialect.
   Tx.ptx.mbarrier.init.shared.b64(...)

.. list-table:: TIRx script layers
   :header-rows: 1
   :widths: 22 22 28 28

   * - Layer
     - Authoring API
     - Python implementation
     - IR constructed
   * - Core TIRx script
     - ``Tx.*``
     - ``tvm.tirx.script.builder.ir``
     - TIRx statements and expressions
   * - Tile primitives
     - ``Tx.tile.*``
     - ``tvm.tirx.script.tile``
     - ``tvm.tirx.TilePrimitiveCall``
   * - Direct CUDA PTX
     - ``Tx.ptx.*``
     - ``tvm.backend.cuda.ptx``
     - ``tirx.ptx.*`` call

``tvm.script.tirx`` is a syntax and construction frontend for the
``tvm.tirx`` object model, not a separate IR.  Core builders create nodes such
as ``tvm.tirx.For``, ``tvm.tirx.BufferLoad``, and ``tvm.ir.Call`` directly.

The high-level tile facade validates tile-only arguments before delegating to
the raw constructors.  Both entry points create the same
``tvm.tirx.TilePrimitiveCall`` node; the raw constructors are not a second,
lower IR.  During ``tvm.tirx.transform.LowerTIRx``, tile-primitive dispatch
selects and inlines a target implementation.  The cleanup stage then resolves
layouts and execution scopes, leaving lower-level TIRx statements and calls.
It does not convert the module to the separate ``tvm.tir`` object model.  The
direct PTX namespace creates ``tirx.ptx.*`` calls immediately and bypasses tile
dispatch; see :doc:`ptx`.

.. code-block:: text

   Tx.* core builders ─────────────────▶ TIRx statements and expressions ──────┐
   Tx.tile.* facade ─▶ TilePrimitiveCall ─▶ backend dispatch ─────────────────┤
   Tx.ptx.* ────────────────────────────▶ tirx.ptx.* Call ─────────────────────┤
                                                                                ▼
                                                 layout/scope cleanup ─▶ lower-level TIRx
                                                                                │
                                                                                ▼
                                                                     later passes + codegen

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
``Tx.tile.copy(...)`` uses thread scope, while ``Tx.tile.warp.copy(...)``,
``Tx.tile.wg.copy(...)``, and ``Tx.tile.cta.copy(...)`` request wider
cooperation.

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
normally use ``tvm.tirx.script.tile``.  Its public builder spelling is
``tvm.script.ir_builder.tirx.tile``::

   from tvm.script.ir_builder import tirx as Tx_builder

   Tx_builder.tile.copy(dst, src)

Every callable tile operation documented in the preceding section has a
same-name raw constructor.  The signatures and docstrings come from these raw
constructors, so expanding them a second time would duplicate the complete API
reference.

.. autoclass:: tvm.tirx.script.builder.tirx.ScopedOp
   :members:
   :no-index:

.. autoclass:: tvm.tirx.script.builder.tirx.ScopeNamespace
   :members:
   :no-index:

.. _tirx-api-dispatch-extension-points:

Dispatch extension points
-------------------------
.. automodule:: tvm.tirx.operator.tile_primitive
   :members:
   :imported-members:
   :no-index:

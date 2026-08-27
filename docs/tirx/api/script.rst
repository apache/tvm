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

The high-level tile facade validates tile-only buffer arguments before it
delegates to the lower-level tile-call constructors.  The direct PTX namespace
emits individual backend instructions and does not perform tile dispatch; see
:doc:`ptx`.

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
   :members:
   :no-index:

Low-level tile-call constructors
--------------------------------
``tvm.tirx.script.builder.tirx`` constructs the underlying
``TilePrimitiveCall`` nodes.  It is useful to parser, builder, and extension
authors; kernel code should normally use the validated high-level facade above.

.. automodule:: tvm.tirx.script.builder.tirx
   :members:
   :no-index:
   :exclude-members: ScopeNamespace, ScopedOp

Dispatch extension points
-------------------------
.. automodule:: tvm.tirx.operator.tile_primitive
   :members:
   :imported-members:
   :no-index:

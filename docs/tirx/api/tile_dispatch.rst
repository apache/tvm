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

Tile Dispatch Extension API
===========================

Raw node constructors
---------------------

``tvm.tirx.script.builder.tirx`` constructs the same
``TilePrimitiveCall`` nodes as the validated ``Tx.tile`` facade.  This surface
is intended for parser, builder, and extension authors; kernel code should
normally use the :doc:`tile authoring API <tile>`.  Its public builder spelling
is ``tvm.script.ir_builder.tirx.tile``::

   from tvm.script.ir_builder import tirx as Tx_builder

   Tx_builder.tile.copy(dst, src)

Each callable tile operation has a same-name raw constructor.  The authoring
page provides their signatures without duplicating the full list here.

.. autoclass:: tvm.tirx.script.builder.tirx.ScopedOp
   :members:
   :no-index:

.. autoclass:: tvm.tirx.script.builder.tirx.ScopeNamespace
   :members:
   :no-index:

.. _tirx-api-dispatch-extension-points:

Dispatch registration
---------------------

.. automodule:: tvm.tirx.operator.tile_primitive
   :members:
   :imported-members:
   :no-index:

See :doc:`../arch/tile_dispatch` for the selection algorithm and the registered
CUDA and Trainium variants.

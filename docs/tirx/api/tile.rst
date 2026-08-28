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

Tile Primitive Authoring API
============================

Kernel code calls tile primitives through ``Tx.tile``::

   from tvm.script import tirx as Tx

   Tx.tile.copy(dst, src)
   Tx.tile.wg.gemm_async(accum, lhs, rhs)

The scope namespaces bind an operation to an execution scope.
``Tx.tile.copy(...)`` uses thread scope, while ``Tx.tile.warp.copy(...)``,
``Tx.tile.wg.copy(...)``, and ``Tx.tile.cta.copy(...)`` request wider
cooperation.  The dialect also exposes those scope objects at its root: for
example, ``Tx.cta.copy(...)``
and ``Tx.tile.cta.copy(...)`` construct the same call.  This documentation uses
the explicit ``Tx.tile`` form consistently.  See the
:doc:`programming guide <../tile_primitives>` for the model, primitive catalog,
and dispatch configuration.

.. automodule:: tvm.tirx.script.tile
   :members: compose_op
   :no-index:

Scope namespaces
----------------

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

Operations
----------

``ScopedOp`` objects are callable but are not ordinary Python functions, so
they are listed explicitly.

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

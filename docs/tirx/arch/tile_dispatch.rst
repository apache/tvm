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

Tile Primitive Dispatch
=======================

This chapter documents how unresolved ``TilePrimitiveCall`` nodes are selected
and lowered.  Kernel authors should start with the
:doc:`Tile Primitives programming guide <../tile_primitives>`; extension
authors can find the callable registration interfaces in
:doc:`../api/tile_dispatch`.

Implementation surfaces
-----------------------

The authoritative operation list is the C++ registry
(``src/tirx/op/tirx.cc``, with operations named ``tirx.tile.<name>``).  IR
wrapper classes live in
``python/tvm/tirx/operator/tile_primitive/ops.py``.  Raw
``TilePrimitiveCall`` constructors live in
``python/tvm/tirx/script/builder/tirx.py``, while the validated authoring facade
is in ``python/tvm/tirx/script/tile.py``.  Both Python construction surfaces
produce the same IR node type.

Dispatch pipeline
-----------------

Dispatch runs in the ``tirx.TilePrimitiveDispatch`` pass, the first phase of
``LowerTIRx()``, before layout and execution-scope cleanup.  The C++ mutator
``TilePrimitiveDispatcher`` walks the IR and, for each call:

#. resolves the ``(inter, intra)`` execution split for the call's scope from the
   active set tracked through control flow (``if wg_id == ...``, ``warp_id``,
   and ``Tx.cuda.elect_sync()``);
#. builds a ``DispatchContext`` carrying the target, scope, launch parameters,
   value ranges, and encoded ``inter``/``intra`` maps plus ``scope_kind``;
#. invokes the global FFI hook ``tirx.f_op_dispatcher`` with the call and
   context, which returns a ``PrimFunc``;
#. splices that ``PrimFunc`` body in place of the call and drains side-effect
   callbacks for private allocations and device or host initialization.

If a ``TilePrimitiveCall`` survives lowering, the verifier reports a fatal
error.

Variant selection
-----------------

The Python dispatcher holds a table keyed by ``(Op, target_kind)``.  Backends
register each case with ``register_dispatch``, including its variant name,
priority, predicates, and implementation.  ``run_dispatch(op_call, sctx)``:

#. looks up the primitive and target pair;
#. filters to ``op_call.dispatch`` when the caller explicitly requests a
   variant;
#. sorts candidates by descending priority and then by variant name;
#. evaluates each candidate's predicates and runs the first implementation that
   accepts the call;
#. continues searching when an implementation raises ``DispatchFail``; and
#. reports every rejection reason when no candidate accepts the call.

Dispatch is therefore target-specific, priority-ordered, and predicate-guarded,
with an optional ``dispatch=`` override.  Common predicates validate matching
operand shapes and layouts or require a complete active thread group, preventing
a partial warp or warpgroup from being lowered by an implementation that needs
full participation.

Variants by primitive
---------------------

The following chapters document the registered variants, their selection
conditions, the IR they emit, and the conditions under which they decline.
Their URLs remain under ``tile_primitives`` for compatibility with existing
links.

.. toctree::
   :maxdepth: 1

   ../tile_primitives/copy
   ../tile_primitives/copy_async
   ../tile_primitives/gemm
   ../tile_primitives/gemm_async
   ../tile_primitives/elementwise
   ../tile_primitives/reduction
   ../tile_primitives/permute_layout

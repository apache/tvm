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

Tensor Instruction Lowering
===========================

Backend tensor operations are ordinary ``tvm.ir.Call`` nodes with opaque
side effects and void return types. CUDA and Trainium register their contracts
in ``python/tvm/backend/{cuda,trn}/tensor_instructions.py`` using the shared
``python/tvm/tirx/tensor_instruction.py`` machinery. C++ reflected attribute
types and the native Call validator are in ``src/tirx/op/tile.cc``.

``tirx.TilePrimitiveDispatch`` remains the first phase of ``LowerTIRx``.
The pass recognizes ``Evaluate(Call)`` by the operator's ``TIRxOpCategory``:
``tile_primitive`` or ``tile_composite``. It validates the call, resolves the
scope against the active thread set, constructs a ``DispatchContext``, and
invokes the instruction's registered lowerer. It replaces the Evaluate with
the returned function body and lowers any nested tensor calls.

This preserves launch parameters, inter/intra scope maps, value ranges,
descriptor caches, host initialization, and allocation/initialization
callbacks. Lowering occurs at a statement boundary because one instruction
may require address preparation, layout expansion, and multiple occurrences
of the same core instruction. A residual-call verifier rejects either
category after expansion.

There is one lowerer per instruction identity. Unsupported layouts and scopes
produce an error that includes the instruction, target, scope, and original
cause. There is no cross-instruction priority search or forced variant bag.

The transient Python ``TensorCall`` view supplies semantic operand names to
existing backend emission code. It is not a registered IR node and is never
serialized. Its decoded options are private lowering data. Every expression
in the original IR resides in ``Call.args``; shared traversal does not inspect
attributes for hidden expressions.

The TVMScript printer validates tensor Calls and prints canonical backend
names, tensor regions, nested tuple operands, and static qualifiers. Parsing
reconstructs the fixed signature. Private Trainium workspace allocation
rewrites Call argument slots before instruction lowering.

See :doc:`../tile_primitives` for instruction contracts and caller-owned
algorithms, and :doc:`../api/tile_dispatch` for registration.

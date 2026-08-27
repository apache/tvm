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

Direct PTX Instructions
=======================

The CUDA backend installs a table-driven PTX instruction namespace on the
TIRx TVMScript dialect.  Its spelling depends only on the local import alias::

   from tvm.script import tirx as T
   T.ptx.add.rn.f32(dst, lhs, rhs)

   from tvm.script import tirx as Tx
   Tx.ptx.add.rn.f32(dst, lhs, rhs)

``T.ptx`` and ``Tx.ptx`` above are the same object.  This layer emits a single
PTX instruction directly; it is below the layout-aware tile primitives
(``Tx.*`` with split aliases or ``Tx.tile.*`` with one alias) and does not run
tile-primitive dispatch.

Instruction forms
-----------------
Attribute chains fill PTX modifier slots.  Python keywords have a trailing
underscore, and PTX's ``::`` separator is written as a double underscore::

   T.ptx.ld.acquire.gpu.global_.b32(value, pointer)
   T.ptx.mbarrier.arrive.shared__cluster.b64(barrier, count)

The indexed form preserves exact PTX spelling and is useful for dynamically
assembled instruction names::

   T.ptx["st.weak.shared::cta.b32"](pointer, value)

Destination registers are leading operands, so PTX calls are statements.
Predication is keyword-only::

   T.ptx.ld.global_.b32(value, pointer, pred=predicate, preserve_dst=True)

``T.ptx.pred(value)`` marks a register operand whose PTX type is ``.pred``;
``T.ptx.addr(base, byte_offset)`` forms an immediate-offset address for
instructions that accept one; and ``T.ptx.SINK`` represents PTX's discard
destination ``_``.  The generated ``tvm/script/tirx.pyi`` stub provides editor
completion for these helpers, registered instruction families, and modifier
chains.

PTX versus CUDA helpers
-----------------------
Use ``T.ptx`` for one table-described PTX instruction.  Use ``T.cuda`` for
backend helpers that require multiple statements, C/C++ expressions, descriptor
packing, or other behavior that cannot be represented as one PTX instruction.
``T.ptx_legacy`` only preserves historical spellings needed by compatibility
passes and is not the API for new kernels.

Namespace API
-------------
.. autoclass:: tvm.backend.cuda.ptx.PTXNamespace
   :members:
   :no-index:
   :special-members: __getitem__

.. automodule:: tvm.backend.cuda.ptx
   :members:
   :no-index:
   :exclude-members: PTXNamespace

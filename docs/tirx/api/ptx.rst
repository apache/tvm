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

The CUDA backend makes a table-driven PTX instruction namespace available
when authoring TIRx with TVMScript.  This is an authoring namespace, not a
separate IR dialect.  Its spelling depends on the local import alias::

   from tvm.script import tirx as T
   T.ptx.add.rn.f32(dst, lhs, rhs)

   from tvm.script import tirx as Tx
   Tx.ptx.add.rn.f32(dst, lhs, rhs)

``T.ptx`` and ``Tx.ptx`` above are the same object.  With the conventional
split aliases, however, ``Tx`` names only the tile facade, so direct PTX is
written through ``T``::

   from tvm.script import tirx as T
   from tvm.script.tirx import tile as Tx

   T.ptx.add.rn.f32(dst, lhs, rhs)  # Direct PTX.
   Tx.copy(src, dst)                # Tile primitive.

Each ``T.ptx`` or ``Tx.ptx`` call must match a form registered in
:mod:`tvm.backend.cuda.ptx.table`.  The table validates the instruction family,
modifier combination, operand count, and operand types while constructing one
``tirx.ptx.*`` IR call.  A new instruction is added by extending the table and,
when it needs a new representation, its code-generation support.

This layer is below the layout-aware tile primitives (``Tx.*`` with split
aliases or ``Tx.tile.*`` with one alias) and does not run tile-primitive
dispatch.

Supported instruction families
------------------------------
The current instruction table has 318 entries across the following 94
families.  These are the Python attribute names written after ``T.ptx`` or
``Tx.ptx``; each family accepts only the modifier and operand combinations
declared by its table entries.

.. hlist::
   :columns: 4

   * ``abs``
   * ``activemask``
   * ``add``
   * ``and_``
   * ``applypriority``
   * ``atom``
   * ``bar``
   * ``barrier``
   * ``bfe``
   * ``bfi``
   * ``bfind``
   * ``bmsk``
   * ``brev``
   * ``clusterlaunchcontrol``
   * ``clz``
   * ``cnot``
   * ``copysign``
   * ``cos``
   * ``cp``
   * ``createpolicy``
   * ``cvt``
   * ``cvt_pack``
   * ``cvta``
   * ``discard``
   * ``div``
   * ``dp2a``
   * ``dp4a``
   * ``elect_sync``
   * ``ex2``
   * ``fence``
   * ``fma``
   * ``fns``
   * ``getctarank``
   * ``griddepcontrol``
   * ``isspacep``
   * ``ld``
   * ``ldmatrix``
   * ``ldu``
   * ``lg2``
   * ``lop3``
   * ``mad``
   * ``mad24``
   * ``mapa``
   * ``match``
   * ``max``
   * ``mbarrier``
   * ``min``
   * ``mma``
   * ``mov``
   * ``movmatrix``
   * ``mul``
   * ``mul24``
   * ``multimem_ld_reduce``
   * ``multimem_red``
   * ``multimem_st``
   * ``neg``
   * ``not_``
   * ``or_``
   * ``popc``
   * ``prefetch``
   * ``prefetchu``
   * ``prmt``
   * ``rcp``
   * ``red``
   * ``red_async``
   * ``redux_sync``
   * ``rem``
   * ``rsqrt``
   * ``sad``
   * ``selp``
   * ``set``
   * ``setmaxnreg``
   * ``setp``
   * ``shf``
   * ``shfl_sync``
   * ``shl``
   * ``shr``
   * ``sin``
   * ``slct``
   * ``sqrt``
   * ``st``
   * ``st_async``
   * ``st_bulk``
   * ``stmatrix``
   * ``sub``
   * ``szext``
   * ``tanh``
   * ``tcgen05``
   * ``tensormap_cp_fenceproxy``
   * ``tensormap_replace``
   * ``testp``
   * ``vote_sync``
   * ``wgmma``
   * ``xor``

The generated ``tvm/script/tirx.pyi`` stub provides editor completion for the
valid modifier names and call signatures.  The instruction table is the
complete source of modifier domains, operand layouts, and combination
constraints.

Instruction forms
-----------------
Attribute chains fill PTX modifier slots.  Python keywords have a trailing
underscore, and PTX's ``::`` separator is written as a double underscore::

   T.ptx.ld.acquire.gpu.global_.b32(value, pointer)
   T.ptx.mbarrier.arrive.shared__cluster.b64(barrier, count)

The indexed form accepts the exact PTX spelling of a registered form and uses
the same table lookup::

   T.ptx["st.weak.shared::cta.b32"](pointer, value)

Destination registers are leading operands, so PTX calls are statements.
Predication is keyword-only::

   T.ptx.ld.global_.b32(value, pointer, pred=predicate, preserve_dst=True)

``T.ptx.pred(value)`` marks a register operand whose PTX type is ``.pred``;
``T.ptx.addr(base, byte_offset)`` forms an immediate-offset address for
instructions that accept one; and ``T.ptx.SINK`` represents PTX's sink operand
``_`` in table-marked positions.  The generated ``tvm/script/tirx.pyi`` stub
provides editor completion for these helpers, registered instruction families,
and modifier chains.

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
   :special-members: __getitem__

.. automodule:: tvm.backend.cuda.ptx
   :members:
   :exclude-members: PTXNamespace

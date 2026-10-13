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

Core TVMScript
==============

TIRx kernels use ``tvm.script.tirx`` for the parser and core IR builders::

   from tvm.script import tirx as Tx

   Tx.alloc_tensor(...)

Tile ops and backend-specific namespaces are documented separately in
:doc:`tile`, :doc:`cuda`, and :doc:`ptx`.  For the relationship between these
authoring layers and TIRx IR, see :ref:`tirx-programming-model`.

Parser entry points
-------------------
.. automodule:: tvm.script.tirx
   :members: function, jit, inline, macro
   :undoc-members:
   :no-index:

Core IR builder
---------------

Common statement frames and construction helpers live in
``tvm.script.ir_builder.frame`` and ``tvm.script.ir_builder.stmt``. The TIRx
namespace re-exports them alongside its function, tensor/layout, and execution
scope builders. Native shared statement builders live under
``include/tvm/script/ir_builder`` and ``src/script/ir_builder``; dialect-specific
thread-placement and tensor alias policies remain in the TIRx/S-TIR builders.

.. automodule:: tvm.tirx.script.ir_builder.ir
   :members:
   :no-index:
   :exclude-members: LetAnnotation, alloc_tcgen05_ldst_frag

.. currentmodule:: tvm.tirx.script.ir_builder.ir

.. py:class:: LetAnnotation(type_spec=None)

   Marker used by ``Tx.let`` and ``Tx.let[dtype]`` annotations to construct an
   explicit ``LetStmt``.

.. py:function:: alloc_tcgen05_ldst_frag(instr_shape, tensor_shape, dtype)

   Allocate a local register fragment whose layout matches a
   ``tcgen05.{ld,st}`` atom. ``instr_shape`` accepts ``"32x32b"``,
   ``"16x64b"``, ``"16x128b"``, or ``"16x256b"``. For example, a
   two-CTA Layout-B accumulator uses this layout, defined outside the kernel::

      from tvm.backend.cuda.layout import tmem_mma_operand_layout

      layout_b = tmem_mma_operand_layout(
          "D", (64, 128), "float32", M=128, cta_group=2)

   Inside the kernel, declare the accumulator over its allocated TMEM address,
   allocate the register fragment, and prepare matching instruction views::

      C = Tx.cuda.decl_tmem(tmem_address, ty_args=[
          Tx.Tensor((64, 128), "float32", scope="tmem", layout=layout_b)
      ])
      frag = Tx.alloc_tcgen05_ldst_frag("32x32b", (64, 128), "float32")
      C_words = C.rearrange("(wh r) (wl c) -> (wl wh) r c",
                            wh=2, r=32, wl=2)
      frag_words = frag.rearrange("(wh r) (wl c) -> (wl wh) r c",
                                  wh=2, r=32, wl=2)
      Tx.cuda.tile.apply(
          Tx.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"],
          frag_words, C_words, scope="warpgroup")
      Tx.ptx.tcgen05.wait__ld.sync.aligned()

   ``tmem_address`` must name a live allocation with at least 64 columns.
   The readback requires a full active warpgroup.
   Complete the producing MMA's commit/mbarrier wait and
   the required ``tcgen05.fence::after_thread_sync`` before the load.

   Layout B stores logical ``C[i,j]`` in TMEM lane
   ``i + 64*(j//64)`` and column ``j%64``. The views above both have shape
   ``(4,32,64)``: ``warp=2*wl+wh`` chooses the physical 32-lane slab,
   ``r`` selects its row, and ``c`` selects its 64 columns. Each warp issues
   one ``.32x32b.x64`` instruction and each lane receives 64 registers.
   No temporary copy is needed. ``wait::ld`` makes the loaded registers
   available to subsequent instructions; it does not wait for an earlier MMA.

   See :doc:`../tile_ops/tcgen05_ld_st` for other native shapes,
   pack/unpack, and a complete store/load example.

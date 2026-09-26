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

   Tx.alloc_buffer(...)

Tile primitives and backend-specific namespaces are documented separately in
:doc:`tile`, :doc:`cuda`, and :doc:`ptx`.  For the relationship between these
authoring layers and TIRx IR, see :ref:`tirx-programming-model`.

Parser entry points
-------------------
.. automodule:: tvm.script.tirx
   :members: prim_func, jit, inline, macro
   :undoc-members:
   :no-index:

Core IR builder
---------------
.. automodule:: tvm.tirx.script.ir_builder.ir
   :members:
   :no-index:
   :exclude-members: LetAnnotation, alloc_tcgen05_ldst_frag, anylist_getitem, anylist_resetitem, anylist_setitem_call_packed, anylist_setitem_call_cpacked

.. currentmodule:: tvm.tirx.script.ir_builder.ir

.. py:class:: LetAnnotation(type_spec=None)

   Marker used by ``Tx.let`` and ``Tx.let[dtype]`` annotations to construct an
   explicit ``LetStmt``.

.. py:function:: alloc_tcgen05_ldst_frag(instr_shape, tensor_shape, dtype)

   Allocate a local register fragment whose layout matches a
   ``tcgen05.{ld,st}`` atom. ``instr_shape`` accepts ``"32x32b"``,
   ``"16x64b"``, ``"16x128b"``, or ``"16x256b"``. For example, a
   two-CTA Layout-B accumulator and its readback fragment can be allocated as::

      C = tmem_pool.alloc_tcgen05_mma_D(
          (64, 128), "float32", M=128, cta_group=2)
      frag = Tx.alloc_tcgen05_ldst_frag("32x32b", (64, 128), "float32")
      Tx.tile.wg.copy_async(frag[:, :], C[:, :])

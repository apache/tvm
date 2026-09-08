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
.. automodule:: tvm.tirx.script.parser.entry
   :members:
   :no-index:
   :exclude-members: TIRInline

Core IR builder
---------------
.. automodule:: tvm.tirx.script.builder.ir
   :members:
   :no-index:
   :exclude-members: LetAnnotation, alloc_tcgen05_ldst_frag, anylist_getitem, anylist_resetitem, anylist_setitem_call_packed, anylist_setitem_call_cpacked, match_buffer

.. currentmodule:: tvm.tirx.script.builder.ir

.. py:function:: match_buffer(param, shape=None, dtype="float32", data=None, strides=None, elem_offset=None, scope="global", align=-1, offset_factor=0, layout="default")

   Bind a function parameter or an existing buffer region to a TIRx buffer.
   ``shape`` is required for a function parameter and is inferred from a
   ``BufferRegion`` when omitted.  ``layout`` accepts a layout object, a
   registered layout string, or ``None``.

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

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
   :exclude-members: anylist_getitem, anylist_resetitem, anylist_setitem_call_packed, anylist_setitem_call_cpacked

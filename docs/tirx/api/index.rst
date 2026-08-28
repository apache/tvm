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

API Reference
=============

Python APIs for the TIRx IR, TVMScript dialect, compilation pipeline, and
backend extension points.  The examples use one ``Tx`` alias: core TIRx is
``Tx.*``, tile primitives are ``Tx.tile.*``, and CUDA's direct instruction
dialect is ``Tx.ptx.*``::

   from tvm.script import tirx as Tx

   # Tx.alloc_buffer(...)
   # Tx.tile.copy(...)
   # Tx.ptx.mbarrier.init.shared.b64(...)

.. toctree::
   :maxdepth: 1

   tirx
   layout
   execution
   script
   ptx
   compilation
   analysis
   stmt_functor
   transform
   backend

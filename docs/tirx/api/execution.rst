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

Execution Scope IR
==================

tvm.tirx.exec_scope
-------------------
.. automodule:: tvm.tirx.exec_scope
   :members:
   :no-index:

Backend scopes
--------------

Execution scope identities remain part of generic layout IR. Backends register
their finite execution hierarchies and provide the user-facing constructors:
``T.cuda.ExecScope("warp")`` and ``T.trn.ExecScope("thread")``.
``T.cuda.ExecScope("kernel")`` names the root used by grid axes such as ``bx``
in the ``kernel -> cta`` relationship. Every scope uniformly describes a
hierarchy node; each backend tile instruction declares the scopes it supports.
For example, CUDA TMA instructions accept ``thread`` and ``tcgen05.ld/st`` accept
``warpgroup``. CUDA index bindings and predicate analysis live in
``tvm.backend.cuda.tile.exec_context``.

.. automodule:: tvm.backend.cuda.exec_scope
   :members:
   :no-index:

.. automodule:: tvm.backend.trn.exec_scope
   :members:
   :no-index:

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

tvm.ir
------
.. automodule:: tvm.ir
   :members:
   :imported-members:
   :exclude-members: Expr
   :autosummary:

Primitive construction
~~~~~~~~~~~~~~~~~~~~~~

Primitive operators eagerly fold constants and identities by default. Use
:py:class:`tvm.ir.prim.OpConstFoldScope` to retain expression structure while
constructing IR. The setting is local to the current thread and is restored on
scope exit, including exceptions. Explicit symbolic simplification remains
available inside a disabled scope.

Type promotion, typed literal construction, and shape normalization remain active.
Type-directed intrinsic results, such as rounding an integer or taking the absolute
value of an unsigned integer, also retain their usual behavior. Explicit symbolic
simplification and interval analysis remain available inside disabled scopes.

TVMScript function frames disable eager folding during signature and body
construction. Ordinary Python arithmetic such as ``1 + 2`` still evaluates in
Python; use IR constants to construct an expression, such as
``T.int32(1) + T.int32(2)``.

.. autoclass:: tvm.ir.prim.OpConstFoldScope

.. autofunction:: tvm.ir.prim.op_const_fold_enabled

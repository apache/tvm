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

tvm.tirx visitors and mutators
==============================

``expr_functor`` and ``stmt_functor`` provide lightweight Python traversal
utilities.  ``functor`` contains the FFI-backed ``PyStmtExprVisitor`` and
``PyStmtExprMutator`` extension classes.  Their similarly named methods serve
different base classes; they are not duplicate aliases.

tvm.tirx.expr_functor
---------------------
.. automodule:: tvm.tirx.expr_functor
   :members:
   :no-index:

tvm.tirx.stmt_functor
---------------------
.. automodule:: tvm.tirx.stmt_functor
   :members:

tvm.tirx.functor
----------------
.. automodule:: tvm.tirx.functor
   :members:
   :no-index:

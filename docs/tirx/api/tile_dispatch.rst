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

Tensor Instruction Registration
===============================

Backend authors define an ``Instruction`` with its canonical operator name,
ordered ``Operand`` list, reflected static attribute schema, and one lowerer.
The shared implementation is ``tvm.tirx.tensor_instruction``; CUDA and
Trainium's ``tensor_instructions.py`` files contain the concrete contracts.

Registration installs the fixed argument count, void return type, opaque
side effects, category, printer name, and native Call validator on the Op.
The validator also runs for directly constructed Calls and printer entry.
Runtime expressions must be operands. New static qualifiers require a typed
attribute field rather than an arbitrary dictionary.

Lowerers receive a semantic view and ``DispatchContext`` and return a
``tvm.tirx.Function`` body. They can prepare pointers/descriptors and expand
layouts for the selected core instruction. They must report unsupported
contracts rather than select a different instruction family.

Named workspace operands have fixed optional slots. Declare a workspace policy
only for instructions that own such scratch storage; do not use it to hide
caller-owned algorithms. See :doc:`../arch/tile_dispatch` for callbacks and
the statement replacement boundary.

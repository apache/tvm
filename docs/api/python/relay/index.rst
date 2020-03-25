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

tvm.relay
=========

This document contains the Python API for the Relay frontend, optimizer, and
compiler toolchain.

Relay is the second-generation, high-level intermediate representation (IR) for the TVM
compiler stack.

.. toctree::
   :maxdepth: 2

   analysis
   backend
   frontend
   image
   transform
   nn
   op
   vision
   testing

.. automodule:: tvm.relay
    :members:
    :imported-members:
    :exclude-members: RelayExpr, Pass, PassInfo, function_pass, PassContext,
      ModulePass, FunctionPass, Sequential, module_pass, Type, TypeKind,
      TypeVar, GlobalTypeVar, TypeConstraint, FuncType, TupleType, IncompleteType,
      TypeCall, TypeRelation, TensorType, RelayRefType, GlobalVar, SourceName,
      Span, Var, Op, Constructor
    :autosummary:

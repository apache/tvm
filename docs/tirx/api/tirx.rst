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

tvm.tirx
========
This page lists the core IR nodes and scalar operations in the top-level
namespace.  Layouts, execution scopes, visitors, compilation helpers, and
tile-dispatch extensions are documented on their focused pages and excluded
here so the same objects are not expanded twice.

.. automodule:: tvm.tirx
   :members:
   :imported-members:
   :exclude-members: Call, ComposeLayout, DispatchContext, ExecScope, Expr,
      ExprFunctor, Layout, Op, PrimExpr, PyStmtExprMutator, PyStmtExprVisitor,
      ScopeIdDef, TileLayout, Var, build, const, get_default_tir_pipeline,
      get_tir_pipeline, register_tir_pipeline

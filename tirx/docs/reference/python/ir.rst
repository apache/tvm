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

tvm_tirx.ir
===========

.. automodule:: tvm_tirx.ir
   :no-members:

.. currentmodule:: tvm_tirx.ir

Location
--------

.. autosummary::
   :toctree: generated/

   Location
   SourceName
   SourceLoc
   UnknownLoc
   CallSiteLoc

Type
----

.. autosummary::
   :toctree: generated/

   Type
   AnyType
   FuncType
   MissingType
   OpaqueType
   PtrType
   PrimType
   StringType
   TensorRegionType
   TupleType

Expr
----

.. autosummary::
   :toctree: generated/

   Expr
   Call
   IntImm
   FloatImm
   Constant
   GenericConst
   DataTypeImm
   StringImm
   GlobalVar
   LambdaExpr
   StagingExpr
   OpaqueExpr
   TensorLoad
   TensorRegion
   Tuple
   TupleGetItem
   Var
   BaseFunc
   Op

Stmt
----

.. autosummary::
   :toctree: generated/

   Stmt
   SeqStmt
   Bind
   Evaluate
   Return
   If
   For
   While
   Break
   Continue
   AssertStmt
   RegionStmt
   TensorStore

IRModule
--------

.. autosummary::
   :toctree: generated/

   IRModule

Supporting classes
------------------

.. autosummary::
   :toctree: generated/

   Attrs
   DictAttrs
   EnvFunc
   Node
   Range
   GlobalInfo

IR helpers
----------

.. autosummary::
   :toctree: generated/

   const
   make_node
   assert_structural_equal
   load_json
   save_json
   is_prim_expr
   is_prim_var
   reinfer_type
   register_op_attr
   stmt_seq
   stmt_list

Namespaces
----------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.ir.prim <ir.prim>`
     - Primitive expressions and construction helpers shared by TVM IR dialects.
   * - :doc:`tvm_tirx.ir.transform <ir.transform>`
     - Utilities for IR transformations.
   * - :doc:`tvm_tirx.ir.json_compact <ir.json_compact>`
     - Tool to upgrade json from historical versions.
   * - :doc:`tvm_tirx.ir.unique_name_supply <ir.unique_name_supply>`
     - Suppliers that are used to guarantee uniqueness of names.

.. toctree::
   :hidden:
   :maxdepth: 2

   ir.prim
   ir.transform
   ir.json_compact
   ir.unique_name_supply

.. raw:: html

   <span id="tvm-tirx-ir-base"></span>

.. raw:: html

   <span id="tvm-tirx-ir-prim"></span>

.. raw:: html

   <span id="tvm-tirx-ir-transform"></span>

.. raw:: html

   <span id="tvm-tirx-ir-transform-instrument"></span>

.. raw:: html

   <span id="tvm-tirx-ir-unique-name-supply"></span>

.. raw:: html

   <span id="tvm-tirx-ir-json-compact"></span>

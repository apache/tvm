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

Namespaces
----------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   ir.prim
   ir.base
   ir.json_compact
   ir.unique_name_supply

:doc:`Transformation passes <ir.transform>`

Location
--------

.. autosummary::
   :toctree: generated/

   SourceName
   Location
   SourceLoc
   UnknownLoc
   CallSiteLoc

Type
----

.. autosummary::
   :toctree: generated/

   AnyType
   FuncType
   MissingType
   OpaqueType
   PtrType
   PrimType
   StringType
   TensorRegionType
   TupleType
   Type

Expr
----

.. autosummary::
   :toctree: generated/

   Call
   IntImm
   FloatImm
   Constant
   GenericConst
   DataTypeImm
   StringImm
   Expr
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

Operator functions (tvm_tirx.ir.prim)
-------------------------------------

.. currentmodule:: tvm_tirx.ir.prim

.. autosummary::

   abs
   acos
   acosh
   all
   any
   asin
   asinh
   assume
   atan
   atan2
   atanh
   bitwise_and
   bitwise_not
   bitwise_or
   bitwise_xor
   ceil
   ceildiv
   clz
   copysign
   cos
   cosh
   div
   erf
   exp
   exp2
   exp10
   floor
   floordiv
   floormod
   fma
   fmod
   hypot
   if_then_else
   indexdiv
   indexmod
   isfinite
   isinf
   isnan
   ldexp
   likely
   log
   log1p
   log2
   log10
   max
   min
   nearbyint
   nextafter
   popcount
   pow
   power
   round
   rsqrt
   shift_left
   shift_right
   sigmoid
   sin
   sinh
   sqrt
   tan
   tanh
   trunc
   truncdiv
   truncmod
   vscale

.. currentmodule:: tvm_tirx.ir

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

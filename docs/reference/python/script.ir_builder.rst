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

tvm_tirx.script.ir_builder
==========================

.. automodule:: tvm_tirx.script.ir_builder
   :no-members:

.. currentmodule:: tvm_tirx.script.ir_builder

Classes
-------

.. autosummary::
   :toctree: generated/

   AlreadyEmitted
   IRBuilder
   IRModuleFrame

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``Bind``
     - :obj:`tvm_tirx.ir.Bind`
   * - ``Call``
     - :obj:`tvm_tirx.ir.Call`
   * - ``DataTypeImm``
     - :obj:`tvm_tirx.ir.DataTypeImm`
   * - ``FuncType``
     - :obj:`tvm_tirx.ir.FuncType`
   * - ``GenericConst``
     - :obj:`tvm_tirx.ir.GenericConst`
   * - ``MissingType``
     - :obj:`tvm_tirx.ir.MissingType`
   * - ``Op``
     - :obj:`tvm_tirx.ir.Op`
   * - ``PrimType``
     - :obj:`tvm_tirx.ir.PrimType`
   * - ``Range``
     - :obj:`tvm_tirx.ir.Range`
   * - ``StringImm``
     - :obj:`tvm_tirx.ir.StringImm`
   * - ``StringType``
     - :obj:`tvm_tirx.ir.StringType`
   * - ``TensorRegion``
     - :obj:`tvm_tirx.ir.TensorRegion`
   * - ``TensorRegionType``
     - :obj:`tvm_tirx.ir.TensorRegionType`
   * - ``Tuple``
     - :obj:`tvm_tirx.ir.Tuple`
   * - ``Type``
     - :obj:`tvm_tirx.ir.Type`
   * - ``Var``
     - :obj:`tvm_tirx.ir.Var`
   * - ``Missing``
     - :obj:`tvm_tirx.ir.MissingType`
   * - ``NoneType``
     - :obj:`tvm_tirx.ir.NoneType`
   * - ``Any``
     - :obj:`tvm_tirx.ir.AnyType`
   * - ``Opaque``
     - :obj:`tvm_tirx.ir.OpaqueType`
   * - ``Prim``
     - :obj:`tvm_tirx.ir.PrimType`
   * - ``String``
     - :obj:`tvm_tirx.ir.StringType`
   * - ``Ptr``
     - :obj:`tvm_tirx.ir.PtrType`
   * - ``Func``
     - :obj:`tvm_tirx.ir.FuncType`

Functions
---------

.. autosummary::
   :toctree: generated/

   at_
   resolve_global_info_args
   with_at_group_
   Lambda
   constexpr
   dtype
   dynamic
   meta_var
   module_attrs
   module_get_attr
   module_global_infos
   module_set_attr
   decl_function
   ir_module
   module_member_
   resolve_global_info_

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``make_node``
     - :obj:`tvm_tirx.ir.make_node`

Namespaces
----------

.. toctree::
   :maxdepth: 1

   script.ir_builder.frame
   script.ir_builder.stmt
   script.ir_builder.tracing

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

tvm_tirx.script.ir_builder.parser_protocol
==========================================

.. automodule:: tvm_tirx.script.ir_builder.parser_protocol
   :no-members:

.. currentmodule:: tvm_tirx.script.ir_builder.parser_protocol

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``ir_module``
     - :obj:`tvm_tirx.script.ir_builder.ir_module`
   * - ``decl_function``
     - :obj:`tvm_tirx.script.ir_builder.decl_function`

.. autosummary::
   :toctree: generated/

   function_
   arg_
   func_name_
   func_ret_type_

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``check_well_formed_``
     - :obj:`tvm_tirx.script.ir_builder.check_well_formed_`
   * - ``resolve_global_info_``
     - :obj:`tvm_tirx.script.ir_builder.resolve_global_info_`

.. autosummary::
   :toctree: generated/

   call_global_var_

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``module_member_``
     - :obj:`tvm_tirx.script.ir_builder.module_member_`

.. autosummary::
   :toctree: generated/

   bind_
   decl_mutable_cell_
   set_mutable_cell_
   unpack_
   emit_
   setitem_
   setattr_
   if_
   then_
   else_
   for_
   while_
   range_
   break_
   continue_
   return_
   assert_
   if_then_else_
   and_
   or_
   not_
   lt_
   le_
   gt_
   ge_
   eq_
   ne_

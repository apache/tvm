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

tvm_tirx.tirx
=============

.. automodule:: tvm_tirx.tirx
   :no-members:

.. currentmodule:: tvm_tirx.tirx

Namespaces
----------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   tirx.analysis
   tirx.tile
   tirx.script
   tirx.bench
   tirx.compilation_pipeline
   tirx.layout
   tirx.op

:doc:`Transformation passes <tirx.transform>`

Classes
-------

.. autosummary::
   :toctree: generated/

   TensorType
   TensorMapType
   Function
   IndexMap
   CallFFIKernelAttr
   ThreadScope
   TileLayout
   Layout
   ComposeLayout

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``Var``
     - :obj:`tvm_tirx.ir.Var`
   * - ``Cast``
     - :obj:`tvm_tirx.ir.prim.Cast`
   * - ``Add``
     - :obj:`tvm_tirx.ir.prim.Add`
   * - ``Sub``
     - :obj:`tvm_tirx.ir.prim.Sub`
   * - ``Mul``
     - :obj:`tvm_tirx.ir.prim.Mul`
   * - ``Div``
     - :obj:`tvm_tirx.ir.prim.Div`
   * - ``Mod``
     - :obj:`tvm_tirx.ir.prim.Mod`
   * - ``FloorDiv``
     - :obj:`tvm_tirx.ir.prim.FloorDiv`
   * - ``FloorMod``
     - :obj:`tvm_tirx.ir.prim.FloorMod`
   * - ``LShift``
     - :obj:`tvm_tirx.ir.prim.LShift`
   * - ``RShift``
     - :obj:`tvm_tirx.ir.prim.RShift`
   * - ``BitwiseAnd``
     - :obj:`tvm_tirx.ir.prim.BitwiseAnd`
   * - ``BitwiseOr``
     - :obj:`tvm_tirx.ir.prim.BitwiseOr`
   * - ``BitwiseXor``
     - :obj:`tvm_tirx.ir.prim.BitwiseXor`
   * - ``BitwiseNot``
     - :obj:`tvm_tirx.ir.prim.BitwiseNot`
   * - ``Min``
     - :obj:`tvm_tirx.ir.prim.Min`
   * - ``Max``
     - :obj:`tvm_tirx.ir.prim.Max`
   * - ``EQ``
     - :obj:`tvm_tirx.ir.prim.EQ`
   * - ``NE``
     - :obj:`tvm_tirx.ir.prim.NE`
   * - ``LT``
     - :obj:`tvm_tirx.ir.prim.LT`
   * - ``LE``
     - :obj:`tvm_tirx.ir.prim.LE`
   * - ``GT``
     - :obj:`tvm_tirx.ir.prim.GT`
   * - ``GE``
     - :obj:`tvm_tirx.ir.prim.GE`
   * - ``And``
     - :obj:`tvm_tirx.ir.prim.And`
   * - ``Or``
     - :obj:`tvm_tirx.ir.prim.Or`
   * - ``Not``
     - :obj:`tvm_tirx.ir.prim.Not`
   * - ``Select``
     - :obj:`tvm_tirx.ir.prim.Select`
   * - ``Let``
     - :obj:`tvm_tirx.ir.prim.Let`

Functions
---------

.. autosummary::
   :toctree: generated/

   tensor_data_ptr
   is_tensor_var
   renew_def
   call_packed_lowered
   call_cpacked_lowered
   register_intrin_lowering
   call_packed
   call_cpacked
   call_intrin
   call_pure_extern
   call_extern
   call_ffi_kernel
   call_llvm_intrin
   call_llvm_pure_intrin
   stack_alloca
   stack_make_shape
   stack_make_dltensor
   handle_add_byte_offset
   abi_field_get
   abi_field_set
   address_of
   assume_aligned
   undef
   gpu_thread_allreduce
   ptr_byte_offset
   throw_last_error
   cpu_parallel_barrier
   gpu_load_matrix_sync
   gpu_store_matrix_sync
   gpu_mma_sync
   gpu_fill_fragment
   vector_low
   vector_high
   vector_combine
   reinterpret
   isnullptr
   logaddexp
   alloc_workspace
   free_workspace
   get_active_lane_mask
   get_vscale_expr
   gpu_dp4a
   ignore_loop_partition
   compile
   get_tir_pipeline
   get_default_tir_pipeline
   register_tir_pipeline

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``convert``
     - :obj:`tvm_tirx.ir.prim.convert`
   * - ``all``
     - :obj:`tvm_tirx.ir.prim.all`
   * - ``any``
     - :obj:`tvm_tirx.ir.prim.any`
   * - ``min_value``
     - :obj:`tvm_tirx.ir.prim.min_value`
   * - ``max_value``
     - :obj:`tvm_tirx.ir.prim.max_value`
   * - ``assume``
     - :obj:`tvm_tirx.ir.prim.assume`
   * - ``infinity``
     - :obj:`tvm_tirx.ir.prim.infinity`
   * - ``exp``
     - :obj:`tvm_tirx.ir.prim.exp`
   * - ``exp2``
     - :obj:`tvm_tirx.ir.prim.exp2`
   * - ``exp10``
     - :obj:`tvm_tirx.ir.prim.exp10`
   * - ``log``
     - :obj:`tvm_tirx.ir.prim.log`
   * - ``log2``
     - :obj:`tvm_tirx.ir.prim.log2`
   * - ``log10``
     - :obj:`tvm_tirx.ir.prim.log10`
   * - ``log1p``
     - :obj:`tvm_tirx.ir.prim.log1p`
   * - ``ldexp``
     - :obj:`tvm_tirx.ir.prim.ldexp`
   * - ``clz``
     - :obj:`tvm_tirx.ir.prim.clz`
   * - ``sin``
     - :obj:`tvm_tirx.ir.prim.sin`
   * - ``sinh``
     - :obj:`tvm_tirx.ir.prim.sinh`
   * - ``asin``
     - :obj:`tvm_tirx.ir.prim.asin`
   * - ``asinh``
     - :obj:`tvm_tirx.ir.prim.asinh`
   * - ``cos``
     - :obj:`tvm_tirx.ir.prim.cos`
   * - ``cosh``
     - :obj:`tvm_tirx.ir.prim.cosh`
   * - ``acos``
     - :obj:`tvm_tirx.ir.prim.acos`
   * - ``acosh``
     - :obj:`tvm_tirx.ir.prim.acosh`
   * - ``tan``
     - :obj:`tvm_tirx.ir.prim.tan`
   * - ``tanh``
     - :obj:`tvm_tirx.ir.prim.tanh`
   * - ``atan``
     - :obj:`tvm_tirx.ir.prim.atan`
   * - ``atan2``
     - :obj:`tvm_tirx.ir.prim.atan2`
   * - ``atanh``
     - :obj:`tvm_tirx.ir.prim.atanh`
   * - ``bitwise_and``
     - :obj:`tvm_tirx.ir.prim.bitwise_and`
   * - ``bitwise_not``
     - :obj:`tvm_tirx.ir.prim.bitwise_not`
   * - ``bitwise_or``
     - :obj:`tvm_tirx.ir.prim.bitwise_or`
   * - ``bitwise_xor``
     - :obj:`tvm_tirx.ir.prim.bitwise_xor`
   * - ``erf``
     - :obj:`tvm_tirx.ir.prim.erf`
   * - ``sigmoid``
     - :obj:`tvm_tirx.ir.prim.sigmoid`
   * - ``sqrt``
     - :obj:`tvm_tirx.ir.prim.sqrt`
   * - ``rsqrt``
     - :obj:`tvm_tirx.ir.prim.rsqrt`
   * - ``floor``
     - :obj:`tvm_tirx.ir.prim.floor`
   * - ``ceil``
     - :obj:`tvm_tirx.ir.prim.ceil`
   * - ``hypot``
     - :obj:`tvm_tirx.ir.prim.hypot`
   * - ``trunc``
     - :obj:`tvm_tirx.ir.prim.trunc`
   * - ``abs``
     - :obj:`tvm_tirx.ir.prim.abs`
   * - ``round``
     - :obj:`tvm_tirx.ir.prim.round`
   * - ``nextafter``
     - :obj:`tvm_tirx.ir.prim.nextafter`
   * - ``nearbyint``
     - :obj:`tvm_tirx.ir.prim.nearbyint`
   * - ``power``
     - :obj:`tvm_tirx.ir.prim.power`
   * - ``pow``
     - :obj:`tvm_tirx.ir.prim.pow`
   * - ``popcount``
     - :obj:`tvm_tirx.ir.prim.popcount`
   * - ``fmod``
     - :obj:`tvm_tirx.ir.prim.fmod`
   * - ``if_then_else``
     - :obj:`tvm_tirx.ir.prim.if_then_else`
   * - ``likely``
     - :obj:`tvm_tirx.ir.prim.likely`
   * - ``isnan``
     - :obj:`tvm_tirx.ir.prim.isnan`
   * - ``isfinite``
     - :obj:`tvm_tirx.ir.prim.isfinite`
   * - ``isinf``
     - :obj:`tvm_tirx.ir.prim.isinf`
   * - ``copysign``
     - :obj:`tvm_tirx.ir.prim.copysign`
   * - ``div``
     - :obj:`tvm_tirx.ir.prim.div`
   * - ``indexdiv``
     - :obj:`tvm_tirx.ir.prim.indexdiv`
   * - ``indexmod``
     - :obj:`tvm_tirx.ir.prim.indexmod`
   * - ``truncdiv``
     - :obj:`tvm_tirx.ir.prim.truncdiv`
   * - ``truncmod``
     - :obj:`tvm_tirx.ir.prim.truncmod`
   * - ``floordiv``
     - :obj:`tvm_tirx.ir.prim.floordiv`
   * - ``floormod``
     - :obj:`tvm_tirx.ir.prim.floormod`
   * - ``ceildiv``
     - :obj:`tvm_tirx.ir.prim.ceildiv`
   * - ``min``
     - :obj:`tvm_tirx.ir.prim.min`
   * - ``max``
     - :obj:`tvm_tirx.ir.prim.max`
   * - ``shift_left``
     - :obj:`tvm_tirx.ir.prim.shift_left`
   * - ``shift_right``
     - :obj:`tvm_tirx.ir.prim.shift_right`
   * - ``vscale``
     - :obj:`tvm_tirx.ir.prim.vscale`

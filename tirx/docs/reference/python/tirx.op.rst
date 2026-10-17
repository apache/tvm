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

tvm_tirx.tirx.op
================

.. automodule:: tvm_tirx.tirx.op
   :no-members:

.. currentmodule:: tvm_tirx.tirx.op

Classes
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``Call``
     - :obj:`tvm_tirx.ir.Call`
   * - ``Expr``
     - :obj:`tvm_tirx.ir.Expr`
   * - ``IntImm``
     - :obj:`tvm_tirx.ir.IntImm`
   * - ``PrimType``
     - :obj:`tvm_tirx.ir.PrimType`
   * - ``PtrType``
     - :obj:`tvm_tirx.ir.PtrType`
   * - ``TensorLoad``
     - :obj:`tvm_tirx.ir.TensorLoad`
   * - ``TensorRegion``
     - :obj:`tvm_tirx.ir.TensorRegion`
   * - ``Var``
     - :obj:`tvm_tirx.ir.Var`
   * - ``Location``
     - :obj:`tvm_tirx.ir.Location`
   * - ``TensorMapType``
     - :obj:`tvm_tirx.tirx.TensorMapType`
   * - ``CallFFIKernelAttr``
     - :obj:`tvm_tirx.tirx.CallFFIKernelAttr`

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``const``
     - :obj:`tvm_tirx.ir.const`
   * - ``clz``
     - :obj:`tvm_tirx.ir.prim.clz`
   * - ``max_value``
     - :obj:`tvm_tirx.ir.prim.max_value`
   * - ``min_value``
     - :obj:`tvm_tirx.ir.prim.min_value`
   * - ``abs``
     - :obj:`tvm_tirx.ir.prim.abs`
   * - ``acos``
     - :obj:`tvm_tirx.ir.prim.acos`
   * - ``acosh``
     - :obj:`tvm_tirx.ir.prim.acosh`
   * - ``all``
     - :obj:`tvm_tirx.ir.prim.all`
   * - ``any``
     - :obj:`tvm_tirx.ir.prim.any`
   * - ``asin``
     - :obj:`tvm_tirx.ir.prim.asin`
   * - ``asinh``
     - :obj:`tvm_tirx.ir.prim.asinh`
   * - ``assume``
     - :obj:`tvm_tirx.ir.prim.assume`
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
   * - ``ceil``
     - :obj:`tvm_tirx.ir.prim.ceil`
   * - ``ceildiv``
     - :obj:`tvm_tirx.ir.prim.ceildiv`
   * - ``copysign``
     - :obj:`tvm_tirx.ir.prim.copysign`
   * - ``cos``
     - :obj:`tvm_tirx.ir.prim.cos`
   * - ``cosh``
     - :obj:`tvm_tirx.ir.prim.cosh`
   * - ``div``
     - :obj:`tvm_tirx.ir.prim.div`
   * - ``erf``
     - :obj:`tvm_tirx.ir.prim.erf`
   * - ``exp``
     - :obj:`tvm_tirx.ir.prim.exp`
   * - ``exp2``
     - :obj:`tvm_tirx.ir.prim.exp2`
   * - ``exp10``
     - :obj:`tvm_tirx.ir.prim.exp10`
   * - ``floor``
     - :obj:`tvm_tirx.ir.prim.floor`
   * - ``floordiv``
     - :obj:`tvm_tirx.ir.prim.floordiv`
   * - ``floormod``
     - :obj:`tvm_tirx.ir.prim.floormod`
   * - ``fma``
     - :obj:`tvm_tirx.ir.prim.fma`
   * - ``fmod``
     - :obj:`tvm_tirx.ir.prim.fmod`
   * - ``hypot``
     - :obj:`tvm_tirx.ir.prim.hypot`
   * - ``if_then_else``
     - :obj:`tvm_tirx.ir.prim.if_then_else`
   * - ``indexdiv``
     - :obj:`tvm_tirx.ir.prim.indexdiv`
   * - ``indexmod``
     - :obj:`tvm_tirx.ir.prim.indexmod`
   * - ``infinity``
     - :obj:`tvm_tirx.ir.prim.infinity`
   * - ``isfinite``
     - :obj:`tvm_tirx.ir.prim.isfinite`
   * - ``isinf``
     - :obj:`tvm_tirx.ir.prim.isinf`
   * - ``isnan``
     - :obj:`tvm_tirx.ir.prim.isnan`
   * - ``ldexp``
     - :obj:`tvm_tirx.ir.prim.ldexp`
   * - ``likely``
     - :obj:`tvm_tirx.ir.prim.likely`
   * - ``log``
     - :obj:`tvm_tirx.ir.prim.log`
   * - ``log1p``
     - :obj:`tvm_tirx.ir.prim.log1p`
   * - ``log2``
     - :obj:`tvm_tirx.ir.prim.log2`
   * - ``log10``
     - :obj:`tvm_tirx.ir.prim.log10`
   * - ``max``
     - :obj:`tvm_tirx.ir.prim.max`
   * - ``min``
     - :obj:`tvm_tirx.ir.prim.min`
   * - ``nearbyint``
     - :obj:`tvm_tirx.ir.prim.nearbyint`
   * - ``nextafter``
     - :obj:`tvm_tirx.ir.prim.nextafter`
   * - ``popcount``
     - :obj:`tvm_tirx.ir.prim.popcount`
   * - ``pow``
     - :obj:`tvm_tirx.ir.prim.pow`
   * - ``power``
     - :obj:`tvm_tirx.ir.prim.power`
   * - ``round``
     - :obj:`tvm_tirx.ir.prim.round`
   * - ``rsqrt``
     - :obj:`tvm_tirx.ir.prim.rsqrt`
   * - ``shift_left``
     - :obj:`tvm_tirx.ir.prim.shift_left`
   * - ``shift_right``
     - :obj:`tvm_tirx.ir.prim.shift_right`
   * - ``sigmoid``
     - :obj:`tvm_tirx.ir.prim.sigmoid`
   * - ``sin``
     - :obj:`tvm_tirx.ir.prim.sin`
   * - ``sinh``
     - :obj:`tvm_tirx.ir.prim.sinh`
   * - ``sqrt``
     - :obj:`tvm_tirx.ir.prim.sqrt`
   * - ``tan``
     - :obj:`tvm_tirx.ir.prim.tan`
   * - ``tanh``
     - :obj:`tvm_tirx.ir.prim.tanh`
   * - ``trunc``
     - :obj:`tvm_tirx.ir.prim.trunc`
   * - ``truncdiv``
     - :obj:`tvm_tirx.ir.prim.truncdiv`
   * - ``truncmod``
     - :obj:`tvm_tirx.ir.prim.truncmod`
   * - ``vscale``
     - :obj:`tvm_tirx.ir.prim.vscale`
   * - ``is_tensor_var``
     - :obj:`tvm_tirx.tirx.is_tensor_var`
   * - ``tensor_data_ptr``
     - :obj:`tvm_tirx.tirx.tensor_data_ptr`
   * - ``register_intrin_lowering``
     - :obj:`tvm_tirx.tirx.register_intrin_lowering`
   * - ``call_packed_lowered``
     - :obj:`tvm_tirx.tirx.call_packed_lowered`
   * - ``call_cpacked_lowered``
     - :obj:`tvm_tirx.tirx.call_cpacked_lowered`
   * - ``call_packed``
     - :obj:`tvm_tirx.tirx.call_packed`
   * - ``call_ffi_kernel``
     - :obj:`tvm_tirx.tirx.call_ffi_kernel`
   * - ``call_cpacked``
     - :obj:`tvm_tirx.tirx.call_cpacked`
   * - ``call_intrin``
     - :obj:`tvm_tirx.tirx.call_intrin`
   * - ``call_pure_extern``
     - :obj:`tvm_tirx.tirx.call_pure_extern`
   * - ``call_extern``
     - :obj:`tvm_tirx.tirx.call_extern`
   * - ``call_llvm_intrin``
     - :obj:`tvm_tirx.tirx.call_llvm_intrin`
   * - ``call_llvm_pure_intrin``
     - :obj:`tvm_tirx.tirx.call_llvm_pure_intrin`
   * - ``stack_alloca``
     - :obj:`tvm_tirx.tirx.stack_alloca`
   * - ``stack_make_shape``
     - :obj:`tvm_tirx.tirx.stack_make_shape`
   * - ``stack_make_dltensor``
     - :obj:`tvm_tirx.tirx.stack_make_dltensor`
   * - ``assume_aligned``
     - :obj:`tvm_tirx.tirx.assume_aligned`
   * - ``undef``
     - :obj:`tvm_tirx.tirx.undef`
   * - ``handle_add_byte_offset``
     - :obj:`tvm_tirx.tirx.handle_add_byte_offset`
   * - ``abi_field_get``
     - :obj:`tvm_tirx.tirx.abi_field_get`
   * - ``abi_field_set``
     - :obj:`tvm_tirx.tirx.abi_field_set`
   * - ``address_of``
     - :obj:`tvm_tirx.tirx.address_of`
   * - ``gpu_thread_allreduce``
     - :obj:`tvm_tirx.tirx.gpu_thread_allreduce`

.. autosummary::
   :toctree: generated/

   gpu_thread_invariant
   gpu_storage_sync

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``cpu_parallel_barrier``
     - :obj:`tvm_tirx.tirx.cpu_parallel_barrier`

.. autosummary::
   :toctree: generated/

   kernel_replace_point
   gpu_warp_shuffle
   gpu_warp_shuffle_up
   gpu_warp_shuffle_down
   gpu_warp_shuffle_xor
   gpu_warp_activemask

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``ptr_byte_offset``
     - :obj:`tvm_tirx.tirx.ptr_byte_offset`
   * - ``throw_last_error``
     - :obj:`tvm_tirx.tirx.throw_last_error`
   * - ``vector_low``
     - :obj:`tvm_tirx.tirx.vector_low`
   * - ``vector_high``
     - :obj:`tvm_tirx.tirx.vector_high`
   * - ``vector_combine``
     - :obj:`tvm_tirx.tirx.vector_combine`
   * - ``gpu_dp4a``
     - :obj:`tvm_tirx.tirx.gpu_dp4a`
   * - ``reinterpret``
     - :obj:`tvm_tirx.tirx.reinterpret`

.. autosummary::
   :toctree: generated/

   gpu_thread_filter
   gpu_active_thread_selector

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``isnullptr``
     - :obj:`tvm_tirx.tirx.isnullptr`
   * - ``logaddexp``
     - :obj:`tvm_tirx.tirx.logaddexp`
   * - ``alloc_workspace``
     - :obj:`tvm_tirx.tirx.alloc_workspace`
   * - ``free_workspace``
     - :obj:`tvm_tirx.tirx.free_workspace`
   * - ``get_active_lane_mask``
     - :obj:`tvm_tirx.tirx.get_active_lane_mask`

.. autosummary::
   :toctree: generated/

   masked_load
   masked_store

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``get_vscale_expr``
     - :obj:`tvm_tirx.tirx.get_vscale_expr`
   * - ``ignore_loop_partition``
     - :obj:`tvm_tirx.tirx.ignore_loop_partition`
   * - ``gpu_load_matrix_sync``
     - :obj:`tvm_tirx.tirx.gpu_load_matrix_sync`
   * - ``gpu_mma_sync``
     - :obj:`tvm_tirx.tirx.gpu_mma_sync`
   * - ``gpu_fill_fragment``
     - :obj:`tvm_tirx.tirx.gpu_fill_fragment`
   * - ``gpu_store_matrix_sync``
     - :obj:`tvm_tirx.tirx.gpu_store_matrix_sync`

.. autosummary::
   :toctree: generated/

   gpu_thread_return

Constants and construction helpers
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``UNKNOWN_LOC``
     - :obj:`tvm_tirx.ir.UNKNOWN_LOC`

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
   * - ``TensorMapType``
     - :obj:`tvm_tirx.tirx.TensorMapType`
   * - ``CallFFIKernelAttr``
     - :obj:`tvm_tirx.tirx.CallFFIKernelAttr`

Functions
---------

.. autosummary::
   :toctree: generated/

   gpu_thread_invariant
   gpu_storage_sync
   kernel_replace_point
   gpu_warp_shuffle
   gpu_warp_shuffle_up
   gpu_warp_shuffle_down
   gpu_warp_shuffle_xor
   gpu_warp_activemask
   gpu_thread_filter
   gpu_active_thread_selector
   masked_load
   masked_store
   gpu_thread_return

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
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
   * - ``cpu_parallel_barrier``
     - :obj:`tvm_tirx.tirx.cpu_parallel_barrier`
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

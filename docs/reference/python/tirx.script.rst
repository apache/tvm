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

tvm_tirx.tirx.script
====================

.. automodule:: tvm_tirx.tirx.script
   :no-members:

.. currentmodule:: tvm_tirx.tirx.script

Classes
-------

.. autosummary::
   :toctree: generated/

   Optional
   LetAnnotation

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
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
   * - ``Type``
     - :obj:`tvm_tirx.ir.Type`
   * - ``Var``
     - :obj:`tvm_tirx.ir.Var`
   * - ``AlreadyEmitted``
     - :obj:`tvm_tirx.script.ir_builder.AlreadyEmitted`
   * - ``IRBuilder``
     - :obj:`tvm_tirx.script.ir_builder.IRBuilder`
   * - ``IRModuleFrame``
     - :obj:`tvm_tirx.script.ir_builder.IRModuleFrame`
   * - ``Missing``
     - :obj:`tvm_tirx.ir.MissingType`
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
   * - ``FloatImm``
     - :obj:`tvm_tirx.ir.FloatImm`
   * - ``IntImm``
     - :obj:`tvm_tirx.ir.IntImm`
   * - ``TensorLoad``
     - :obj:`tvm_tirx.ir.TensorLoad`
   * - ``MemoryScope``
     - :obj:`tvm_tirx.runtime.MemoryScope`
   * - ``Axis``
     - :obj:`tvm_tirx.tirx.layout.Axis`
   * - ``ComposeLayout``
     - :obj:`tvm_tirx.tirx.ComposeLayout`
   * - ``Iter``
     - :obj:`tvm_tirx.tirx.layout.Iter`
   * - ``Layout``
     - :obj:`tvm_tirx.tirx.Layout`
   * - ``TileLayout``
     - :obj:`tvm_tirx.tirx.TileLayout`
   * - ``ThreadScope``
     - :obj:`tvm_tirx.tirx.ThreadScope`
   * - ``TensorMap``
     - :obj:`tvm_tirx.tirx.TensorMapType`
   * - ``Tensor``
     - :obj:`tvm_tirx.tirx.TensorType`
   * - ``CallEffectKind``
     - :obj:`tvm_tirx.ir.CallEffectKind`
   * - ``Broadcast``
     - :obj:`tvm_tirx.ir.prim.Broadcast`
   * - ``Ramp``
     - :obj:`tvm_tirx.ir.prim.Ramp`
   * - ``Shuffle``
     - :obj:`tvm_tirx.ir.prim.Shuffle`
   * - ``EQ``
     - :obj:`tvm_tirx.ir.prim.EQ`
   * - ``GE``
     - :obj:`tvm_tirx.ir.prim.GE`
   * - ``GT``
     - :obj:`tvm_tirx.ir.prim.GT`
   * - ``LE``
     - :obj:`tvm_tirx.ir.prim.LE`
   * - ``LT``
     - :obj:`tvm_tirx.ir.prim.LT`
   * - ``NE``
     - :obj:`tvm_tirx.ir.prim.NE`
   * - ``Add``
     - :obj:`tvm_tirx.ir.prim.Add`
   * - ``And``
     - :obj:`tvm_tirx.ir.prim.And`
   * - ``BitwiseAnd``
     - :obj:`tvm_tirx.ir.prim.BitwiseAnd`
   * - ``BitwiseNot``
     - :obj:`tvm_tirx.ir.prim.BitwiseNot`
   * - ``BitwiseOr``
     - :obj:`tvm_tirx.ir.prim.BitwiseOr`
   * - ``BitwiseXor``
     - :obj:`tvm_tirx.ir.prim.BitwiseXor`
   * - ``Cast``
     - :obj:`tvm_tirx.ir.prim.Cast`
   * - ``Div``
     - :obj:`tvm_tirx.ir.prim.Div`
   * - ``FloorDiv``
     - :obj:`tvm_tirx.ir.prim.FloorDiv`
   * - ``FloorMod``
     - :obj:`tvm_tirx.ir.prim.FloorMod`
   * - ``LShift``
     - :obj:`tvm_tirx.ir.prim.LShift`
   * - ``Max``
     - :obj:`tvm_tirx.ir.prim.Max`
   * - ``Min``
     - :obj:`tvm_tirx.ir.prim.Min`
   * - ``Mod``
     - :obj:`tvm_tirx.ir.prim.Mod`
   * - ``Mul``
     - :obj:`tvm_tirx.ir.prim.Mul`
   * - ``Not``
     - :obj:`tvm_tirx.ir.prim.Not`
   * - ``Or``
     - :obj:`tvm_tirx.ir.prim.Or`
   * - ``RShift``
     - :obj:`tvm_tirx.ir.prim.RShift`
   * - ``Select``
     - :obj:`tvm_tirx.ir.prim.Select`
   * - ``Sub``
     - :obj:`tvm_tirx.ir.prim.Sub`
   * - ``CallFFIKernelAttr``
     - :obj:`tvm_tirx.tirx.CallFFIKernelAttr`
   * - ``broadcast``
     - :obj:`tvm_tirx.ir.prim.Broadcast`
   * - ``ramp``
     - :obj:`tvm_tirx.ir.prim.Ramp`

Functions
---------

.. autosummary::
   :toctree: generated/

   Tuple
   check_well_formed_
   resolve_global_info_
   alloc_tensor
   decl_tensor
   alloc_shared
   alloc_local
   tmem
   alloc_scalar
   decl_scalar
   shared_scalar
   local_scalar
   assign
   static_assert
   int8x2
   int16x2
   int32x2
   int64x2
   int8x4
   int16x4
   int32x4
   int64x4
   int8x8
   int16x8
   int32x8
   int64x8
   int8x16
   int16x16
   int32x16
   int64x16
   int8x32
   int16x32
   int32x32
   int64x32
   int8x64
   int16x64
   int32x64
   int64x64
   uint8x2
   uint16x2
   uint32x2
   uint64x2
   uint8x4
   uint16x4
   uint32x4
   uint64x4
   uint8x8
   uint16x8
   uint32x8
   uint64x8
   uint8x16
   uint16x16
   uint32x16
   uint64x16
   uint8x32
   uint16x32
   uint32x32
   uint64x32
   uint8x64
   uint16x64
   uint32x64
   uint64x64
   float16x2
   float32x2
   float64x2
   float16x4
   float32x4
   float64x4
   float16x8
   float32x8
   float64x8
   float16x16
   float32x16
   float64x16
   float16x32
   float32x32
   float64x32
   float16x64
   float32x64
   float64x64
   float8_e3m4
   float8_e3m4x2
   float8_e3m4x4
   float8_e3m4x8
   float8_e3m4x16
   float8_e3m4x32
   float8_e3m4x64
   float8_e4m3
   float8_e4m3x2
   float8_e4m3x4
   float8_e4m3x8
   float8_e4m3x16
   float8_e4m3x32
   float8_e4m3x64
   float8_e4m3b11fnuz
   float8_e4m3b11fnuzx2
   float8_e4m3b11fnuzx4
   float8_e4m3b11fnuzx8
   float8_e4m3b11fnuzx16
   float8_e4m3b11fnuzx32
   float8_e4m3b11fnuzx64
   float8_e4m3fn
   float8_e4m3fnx2
   float8_e4m3fnx4
   float8_e4m3fnx8
   float8_e4m3fnx16
   float8_e4m3fnx32
   float8_e4m3fnx64
   float8_e4m3fnuz
   float8_e4m3fnuzx2
   float8_e4m3fnuzx4
   float8_e4m3fnuzx8
   float8_e4m3fnuzx16
   float8_e4m3fnuzx32
   float8_e4m3fnuzx64
   float8_e5m2
   float8_e5m2x2
   float8_e5m2x4
   float8_e5m2x8
   float8_e5m2x16
   float8_e5m2x32
   float8_e5m2x64
   float8_e5m2fnuz
   float8_e5m2fnuzx2
   float8_e5m2fnuzx4
   float8_e5m2fnuzx8
   float8_e5m2fnuzx16
   float8_e5m2fnuzx32
   float8_e5m2fnuzx64
   float8_e8m0fnu
   float8_e8m0fnux2
   float8_e8m0fnux4
   float8_e8m0fnux8
   float8_e8m0fnux16
   float8_e8m0fnux32
   float8_e8m0fnux64
   float6_e2m3fn
   float6_e2m3fnx2
   float6_e2m3fnx4
   float6_e2m3fnx8
   float6_e2m3fnx16
   float6_e2m3fnx32
   float6_e2m3fnx64
   float6_e3m2fn
   float6_e3m2fnx2
   float6_e3m2fnx4
   float6_e3m2fnx8
   float6_e3m2fnx16
   float6_e3m2fnx32
   float6_e3m2fnx64
   float4_e2m1fn
   float4_e2m1fnx2
   float4_e2m1fnx4
   float4_e2m1fnx8
   float4_e2m1fnx16
   float4_e2m1fnx32
   float4_e2m1fnx64
   f16
   f32
   f64
   bf16
   i8
   i16
   i32
   i64
   u8
   u16
   u32
   u64
   handle
   void
   ptr
   index_map
   target
   meta_class
   bool
   call_kernel
   cast
   Let
   min
   max
   register_script_namespace
   abs
   acos
   acosh
   address_of
   atomic_add
   call_spirv_pure_glsl450
   device_scope
   prefetch
   static_handle
   asin
   asinh
   atan
   atan2
   atanh
   bitwise_and
   bitwise_not
   bitwise_or
   bitwise_xor
   ceil
   clz
   copysign
   cos
   cosh
   erf
   exp
   exp2
   exp10
   gpu_thread_filter
   gpu_active_thread_selector
   floor
   ceildiv
   floordiv
   floormod
   fmod
   fma
   hypot
   if_then_else
   infinity
   isfinite
   isinf
   isnan
   isnullptr
   ldexp
   likely
   log
   log1p
   log2
   log10
   max_value
   min_value
   nearbyint
   nextafter
   popcount
   pow
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
   gpu_thread_return
   trunc
   truncdiv
   truncmod
   ptr_byte_offset
   throw_last_error
   stack_alloca
   stack_make_shape
   stack_make_dltensor
   call_packed
   call_ffi_kernel
   call_cpacked
   call_packed_lowered
   call_cpacked_lowered
   tensor_data_ptr
   handle_add_byte_offset
   abi_field_set
   abi_field_get
   gpu_thread_invariant
   gpu_thread_allreduce
   gpu_load_matrix_sync
   gpu_mma_sync
   gpu_fill_fragment
   gpu_store_matrix_sync
   gpu_storage_sync
   cpu_parallel_barrier
   kernel_replace_point
   gpu_warp_shuffle
   gpu_warp_shuffle_up
   gpu_warp_shuffle_down
   gpu_warp_shuffle_xor
   gpu_warp_activemask
   assume
   assume_aligned
   undef
   alloc_workspace
   free_workspace
   vscale
   ignore_loop_partition
   reinterpret
   call_extern
   call_intrin
   call_llvm_intrin
   call_llvm_pure_intrin
   call_pure_extern
   vector_low
   vector_high
   vector_combine
   get_active_lane_mask
   masked_load
   masked_store
   gpu_dp4a
   fabs
   logical_and
   logical_or
   logical_not
   select
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
   add_to_parent
   assert_
   break_
   continue_
   else_
   evaluate
   grid
   if_
   region
   return_
   then_
   unpack_
   while_
   function
   function_
   arg_
   func_name_
   func_attr
   call_global_var_
   bind
   bind_
   decl_mutable_cell_
   set_mutable_cell_
   setitem_
   setattr_
   tensor_store
   for_
   range_
   serial
   parallel
   vectorized
   unroll
   device_entry
   device_context
   compute_scope
   parallel_launch
   launch_thread
   func_ret
   emit
   frame_scope
   tensor_indices
   inline
   macro
   jit

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``make_node``
     - :obj:`tvm_tirx.ir.make_node`
   * - ``at_``
     - :obj:`tvm_tirx.script.ir_builder.at_`
   * - ``resolve_global_info_args``
     - :obj:`tvm_tirx.script.ir_builder.resolve_global_info_args`
   * - ``with_at_group_``
     - :obj:`tvm_tirx.script.ir_builder.with_at_group_`
   * - ``Lambda``
     - :obj:`tvm_tirx.script.ir_builder.Lambda`
   * - ``constexpr``
     - :obj:`tvm_tirx.script.ir_builder.constexpr`
   * - ``dtype``
     - :obj:`tvm_tirx.script.ir_builder.dtype`
   * - ``dynamic``
     - :obj:`tvm_tirx.script.ir_builder.dynamic`
   * - ``meta_var``
     - :obj:`tvm_tirx.script.ir_builder.meta_var`
   * - ``module_attrs``
     - :obj:`tvm_tirx.script.ir_builder.module_attrs`
   * - ``module_get_attr``
     - :obj:`tvm_tirx.script.ir_builder.module_get_attr`
   * - ``module_global_infos``
     - :obj:`tvm_tirx.script.ir_builder.module_global_infos`
   * - ``module_set_attr``
     - :obj:`tvm_tirx.script.ir_builder.module_set_attr`
   * - ``decl_function``
     - :obj:`tvm_tirx.script.ir_builder.decl_function`
   * - ``ir_module``
     - :obj:`tvm_tirx.script.parser.ir_module`
   * - ``module_member_``
     - :obj:`tvm_tirx.script.ir_builder.module_member_`
   * - ``smem``
     - :obj:`tvm_tirx.tirx.script.alloc_shared`
   * - ``int8``
     - :obj:`tvm_tirx.tirx.script.i8`
   * - ``int16``
     - :obj:`tvm_tirx.tirx.script.i16`
   * - ``int32``
     - :obj:`tvm_tirx.tirx.script.i32`
   * - ``int64``
     - :obj:`tvm_tirx.tirx.script.i64`
   * - ``uint8``
     - :obj:`tvm_tirx.tirx.script.u8`
   * - ``uint16``
     - :obj:`tvm_tirx.tirx.script.u16`
   * - ``uint32``
     - :obj:`tvm_tirx.tirx.script.u32`
   * - ``uint64``
     - :obj:`tvm_tirx.tirx.script.u64`
   * - ``float16``
     - :obj:`tvm_tirx.tirx.script.f16`
   * - ``float32``
     - :obj:`tvm_tirx.tirx.script.f32`
   * - ``float64``
     - :obj:`tvm_tirx.tirx.script.f64`
   * - ``bfloat16``
     - :obj:`tvm_tirx.tirx.script.bf16`
   * - ``boolean``
     - :obj:`tvm_tirx.tirx.script.bool`
   * - ``llvm_lookup_intrinsic_id``
     - :obj:`tvm_tirx.codegen.llvm.lookup_intrinsic_id`
   * - ``func_ret_type_``
     - :obj:`tvm_tirx.tirx.script.func_ret`
   * - ``emit_``
     - :obj:`tvm_tirx.tirx.script.emit`
   * - ``pyfunc``
     - :obj:`tvm_tirx.script.parser.pyfunc`

Namespaces
----------

.. toctree::
   :maxdepth: 1

   tirx.script.ir_builder
   tirx.script.jit
   tirx.script.nvshmem
   tirx.script.ptx
   tirx.script.ptx_legacy

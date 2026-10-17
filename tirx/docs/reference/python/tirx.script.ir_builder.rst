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

tvm_tirx.tirx.script.ir_builder
===============================

.. automodule:: tvm_tirx.tirx.script.ir_builder
   :no-members:

.. currentmodule:: tvm_tirx.tirx.script.ir_builder

Classes
-------

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
   * - ``Any``
     - :obj:`tvm_tirx.ir.AnyType`
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
   * - ``LetAnnotation``
     - :obj:`tvm_tirx.tirx.script.LetAnnotation`
   * - ``TensorMap``
     - :obj:`tvm_tirx.tirx.TensorMapType`
   * - ``Ptr``
     - :obj:`tvm_tirx.ir.PtrType`
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
   * - ``Missing``
     - :obj:`tvm_tirx.ir.MissingType`
   * - ``Opaque``
     - :obj:`tvm_tirx.ir.OpaqueType`
   * - ``Prim``
     - :obj:`tvm_tirx.ir.PrimType`
   * - ``String``
     - :obj:`tvm_tirx.ir.StringType`
   * - ``Func``
     - :obj:`tvm_tirx.ir.FuncType`

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``Tuple``
     - :obj:`tvm_tirx.tirx.script.Tuple`
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
   * - ``check_well_formed_``
     - :obj:`tvm_tirx.tirx.script.check_well_formed_`
   * - ``decl_function``
     - :obj:`tvm_tirx.script.ir_builder.decl_function`
   * - ``ir_module``
     - :obj:`tvm_tirx.script.ir_builder.ir_module`
   * - ``module_member_``
     - :obj:`tvm_tirx.script.ir_builder.module_member_`
   * - ``resolve_global_info_``
     - :obj:`tvm_tirx.tirx.script.resolve_global_info_`
   * - ``alloc_tensor``
     - :obj:`tvm_tirx.tirx.script.alloc_tensor`
   * - ``decl_tensor``
     - :obj:`tvm_tirx.tirx.script.decl_tensor`
   * - ``alloc_shared``
     - :obj:`tvm_tirx.tirx.script.alloc_shared`
   * - ``alloc_local``
     - :obj:`tvm_tirx.tirx.script.alloc_local`
   * - ``smem``
     - :obj:`tvm_tirx.tirx.script.alloc_shared`
   * - ``tmem``
     - :obj:`tvm_tirx.tirx.script.tmem`
   * - ``alloc_scalar``
     - :obj:`tvm_tirx.tirx.script.alloc_scalar`
   * - ``decl_scalar``
     - :obj:`tvm_tirx.tirx.script.decl_scalar`
   * - ``shared_scalar``
     - :obj:`tvm_tirx.tirx.script.shared_scalar`
   * - ``local_scalar``
     - :obj:`tvm_tirx.tirx.script.local_scalar`
   * - ``assign``
     - :obj:`tvm_tirx.tirx.script.assign`
   * - ``static_assert``
     - :obj:`tvm_tirx.tirx.script.static_assert`
   * - ``int8``
     - :obj:`tvm_tirx.tirx.script.i8`
   * - ``int16``
     - :obj:`tvm_tirx.tirx.script.i16`
   * - ``int32``
     - :obj:`tvm_tirx.tirx.script.i32`
   * - ``int64``
     - :obj:`tvm_tirx.tirx.script.i64`
   * - ``int8x2``
     - :obj:`tvm_tirx.tirx.script.int8x2`
   * - ``int16x2``
     - :obj:`tvm_tirx.tirx.script.int16x2`
   * - ``int32x2``
     - :obj:`tvm_tirx.tirx.script.int32x2`
   * - ``int64x2``
     - :obj:`tvm_tirx.tirx.script.int64x2`
   * - ``int8x4``
     - :obj:`tvm_tirx.tirx.script.int8x4`
   * - ``int16x4``
     - :obj:`tvm_tirx.tirx.script.int16x4`
   * - ``int32x4``
     - :obj:`tvm_tirx.tirx.script.int32x4`
   * - ``int64x4``
     - :obj:`tvm_tirx.tirx.script.int64x4`
   * - ``int8x8``
     - :obj:`tvm_tirx.tirx.script.int8x8`
   * - ``int16x8``
     - :obj:`tvm_tirx.tirx.script.int16x8`
   * - ``int32x8``
     - :obj:`tvm_tirx.tirx.script.int32x8`
   * - ``int64x8``
     - :obj:`tvm_tirx.tirx.script.int64x8`
   * - ``int8x16``
     - :obj:`tvm_tirx.tirx.script.int8x16`
   * - ``int16x16``
     - :obj:`tvm_tirx.tirx.script.int16x16`
   * - ``int32x16``
     - :obj:`tvm_tirx.tirx.script.int32x16`
   * - ``int64x16``
     - :obj:`tvm_tirx.tirx.script.int64x16`
   * - ``int8x32``
     - :obj:`tvm_tirx.tirx.script.int8x32`
   * - ``int16x32``
     - :obj:`tvm_tirx.tirx.script.int16x32`
   * - ``int32x32``
     - :obj:`tvm_tirx.tirx.script.int32x32`
   * - ``int64x32``
     - :obj:`tvm_tirx.tirx.script.int64x32`
   * - ``int8x64``
     - :obj:`tvm_tirx.tirx.script.int8x64`
   * - ``int16x64``
     - :obj:`tvm_tirx.tirx.script.int16x64`
   * - ``int32x64``
     - :obj:`tvm_tirx.tirx.script.int32x64`
   * - ``int64x64``
     - :obj:`tvm_tirx.tirx.script.int64x64`
   * - ``uint8``
     - :obj:`tvm_tirx.tirx.script.u8`
   * - ``uint16``
     - :obj:`tvm_tirx.tirx.script.u16`
   * - ``uint32``
     - :obj:`tvm_tirx.tirx.script.u32`
   * - ``uint64``
     - :obj:`tvm_tirx.tirx.script.u64`
   * - ``uint8x2``
     - :obj:`tvm_tirx.tirx.script.uint8x2`
   * - ``uint16x2``
     - :obj:`tvm_tirx.tirx.script.uint16x2`
   * - ``uint32x2``
     - :obj:`tvm_tirx.tirx.script.uint32x2`
   * - ``uint64x2``
     - :obj:`tvm_tirx.tirx.script.uint64x2`
   * - ``uint8x4``
     - :obj:`tvm_tirx.tirx.script.uint8x4`
   * - ``uint16x4``
     - :obj:`tvm_tirx.tirx.script.uint16x4`
   * - ``uint32x4``
     - :obj:`tvm_tirx.tirx.script.uint32x4`
   * - ``uint64x4``
     - :obj:`tvm_tirx.tirx.script.uint64x4`
   * - ``uint8x8``
     - :obj:`tvm_tirx.tirx.script.uint8x8`
   * - ``uint16x8``
     - :obj:`tvm_tirx.tirx.script.uint16x8`
   * - ``uint32x8``
     - :obj:`tvm_tirx.tirx.script.uint32x8`
   * - ``uint64x8``
     - :obj:`tvm_tirx.tirx.script.uint64x8`
   * - ``uint8x16``
     - :obj:`tvm_tirx.tirx.script.uint8x16`
   * - ``uint16x16``
     - :obj:`tvm_tirx.tirx.script.uint16x16`
   * - ``uint32x16``
     - :obj:`tvm_tirx.tirx.script.uint32x16`
   * - ``uint64x16``
     - :obj:`tvm_tirx.tirx.script.uint64x16`
   * - ``uint8x32``
     - :obj:`tvm_tirx.tirx.script.uint8x32`
   * - ``uint16x32``
     - :obj:`tvm_tirx.tirx.script.uint16x32`
   * - ``uint32x32``
     - :obj:`tvm_tirx.tirx.script.uint32x32`
   * - ``uint64x32``
     - :obj:`tvm_tirx.tirx.script.uint64x32`
   * - ``uint8x64``
     - :obj:`tvm_tirx.tirx.script.uint8x64`
   * - ``uint16x64``
     - :obj:`tvm_tirx.tirx.script.uint16x64`
   * - ``uint32x64``
     - :obj:`tvm_tirx.tirx.script.uint32x64`
   * - ``uint64x64``
     - :obj:`tvm_tirx.tirx.script.uint64x64`
   * - ``float16``
     - :obj:`tvm_tirx.tirx.script.f16`
   * - ``float32``
     - :obj:`tvm_tirx.tirx.script.f32`
   * - ``float64``
     - :obj:`tvm_tirx.tirx.script.f64`
   * - ``float16x2``
     - :obj:`tvm_tirx.tirx.script.float16x2`
   * - ``float32x2``
     - :obj:`tvm_tirx.tirx.script.float32x2`
   * - ``float64x2``
     - :obj:`tvm_tirx.tirx.script.float64x2`
   * - ``float16x4``
     - :obj:`tvm_tirx.tirx.script.float16x4`
   * - ``float32x4``
     - :obj:`tvm_tirx.tirx.script.float32x4`
   * - ``float64x4``
     - :obj:`tvm_tirx.tirx.script.float64x4`
   * - ``float16x8``
     - :obj:`tvm_tirx.tirx.script.float16x8`
   * - ``float32x8``
     - :obj:`tvm_tirx.tirx.script.float32x8`
   * - ``float64x8``
     - :obj:`tvm_tirx.tirx.script.float64x8`
   * - ``float16x16``
     - :obj:`tvm_tirx.tirx.script.float16x16`
   * - ``float32x16``
     - :obj:`tvm_tirx.tirx.script.float32x16`
   * - ``float64x16``
     - :obj:`tvm_tirx.tirx.script.float64x16`
   * - ``float16x32``
     - :obj:`tvm_tirx.tirx.script.float16x32`
   * - ``float32x32``
     - :obj:`tvm_tirx.tirx.script.float32x32`
   * - ``float64x32``
     - :obj:`tvm_tirx.tirx.script.float64x32`
   * - ``float16x64``
     - :obj:`tvm_tirx.tirx.script.float16x64`
   * - ``float32x64``
     - :obj:`tvm_tirx.tirx.script.float32x64`
   * - ``float64x64``
     - :obj:`tvm_tirx.tirx.script.float64x64`
   * - ``float8_e3m4``
     - :obj:`tvm_tirx.tirx.script.float8_e3m4`
   * - ``float8_e3m4x2``
     - :obj:`tvm_tirx.tirx.script.float8_e3m4x2`
   * - ``float8_e3m4x4``
     - :obj:`tvm_tirx.tirx.script.float8_e3m4x4`
   * - ``float8_e3m4x8``
     - :obj:`tvm_tirx.tirx.script.float8_e3m4x8`
   * - ``float8_e3m4x16``
     - :obj:`tvm_tirx.tirx.script.float8_e3m4x16`
   * - ``float8_e3m4x32``
     - :obj:`tvm_tirx.tirx.script.float8_e3m4x32`
   * - ``float8_e3m4x64``
     - :obj:`tvm_tirx.tirx.script.float8_e3m4x64`
   * - ``float8_e4m3``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3`
   * - ``float8_e4m3x2``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3x2`
   * - ``float8_e4m3x4``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3x4`
   * - ``float8_e4m3x8``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3x8`
   * - ``float8_e4m3x16``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3x16`
   * - ``float8_e4m3x32``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3x32`
   * - ``float8_e4m3x64``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3x64`
   * - ``float8_e4m3b11fnuz``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3b11fnuz`
   * - ``float8_e4m3b11fnuzx2``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3b11fnuzx2`
   * - ``float8_e4m3b11fnuzx4``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3b11fnuzx4`
   * - ``float8_e4m3b11fnuzx8``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3b11fnuzx8`
   * - ``float8_e4m3b11fnuzx16``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3b11fnuzx16`
   * - ``float8_e4m3b11fnuzx32``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3b11fnuzx32`
   * - ``float8_e4m3b11fnuzx64``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3b11fnuzx64`
   * - ``float8_e4m3fn``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fn`
   * - ``float8_e4m3fnx2``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnx2`
   * - ``float8_e4m3fnx4``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnx4`
   * - ``float8_e4m3fnx8``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnx8`
   * - ``float8_e4m3fnx16``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnx16`
   * - ``float8_e4m3fnx32``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnx32`
   * - ``float8_e4m3fnx64``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnx64`
   * - ``float8_e4m3fnuz``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnuz`
   * - ``float8_e4m3fnuzx2``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnuzx2`
   * - ``float8_e4m3fnuzx4``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnuzx4`
   * - ``float8_e4m3fnuzx8``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnuzx8`
   * - ``float8_e4m3fnuzx16``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnuzx16`
   * - ``float8_e4m3fnuzx32``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnuzx32`
   * - ``float8_e4m3fnuzx64``
     - :obj:`tvm_tirx.tirx.script.float8_e4m3fnuzx64`
   * - ``float8_e5m2``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2`
   * - ``float8_e5m2x2``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2x2`
   * - ``float8_e5m2x4``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2x4`
   * - ``float8_e5m2x8``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2x8`
   * - ``float8_e5m2x16``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2x16`
   * - ``float8_e5m2x32``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2x32`
   * - ``float8_e5m2x64``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2x64`
   * - ``float8_e5m2fnuz``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2fnuz`
   * - ``float8_e5m2fnuzx2``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2fnuzx2`
   * - ``float8_e5m2fnuzx4``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2fnuzx4`
   * - ``float8_e5m2fnuzx8``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2fnuzx8`
   * - ``float8_e5m2fnuzx16``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2fnuzx16`
   * - ``float8_e5m2fnuzx32``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2fnuzx32`
   * - ``float8_e5m2fnuzx64``
     - :obj:`tvm_tirx.tirx.script.float8_e5m2fnuzx64`
   * - ``float8_e8m0fnu``
     - :obj:`tvm_tirx.tirx.script.float8_e8m0fnu`
   * - ``float8_e8m0fnux2``
     - :obj:`tvm_tirx.tirx.script.float8_e8m0fnux2`
   * - ``float8_e8m0fnux4``
     - :obj:`tvm_tirx.tirx.script.float8_e8m0fnux4`
   * - ``float8_e8m0fnux8``
     - :obj:`tvm_tirx.tirx.script.float8_e8m0fnux8`
   * - ``float8_e8m0fnux16``
     - :obj:`tvm_tirx.tirx.script.float8_e8m0fnux16`
   * - ``float8_e8m0fnux32``
     - :obj:`tvm_tirx.tirx.script.float8_e8m0fnux32`
   * - ``float8_e8m0fnux64``
     - :obj:`tvm_tirx.tirx.script.float8_e8m0fnux64`
   * - ``float6_e2m3fn``
     - :obj:`tvm_tirx.tirx.script.float6_e2m3fn`
   * - ``float6_e2m3fnx2``
     - :obj:`tvm_tirx.tirx.script.float6_e2m3fnx2`
   * - ``float6_e2m3fnx4``
     - :obj:`tvm_tirx.tirx.script.float6_e2m3fnx4`
   * - ``float6_e2m3fnx8``
     - :obj:`tvm_tirx.tirx.script.float6_e2m3fnx8`
   * - ``float6_e2m3fnx16``
     - :obj:`tvm_tirx.tirx.script.float6_e2m3fnx16`
   * - ``float6_e2m3fnx32``
     - :obj:`tvm_tirx.tirx.script.float6_e2m3fnx32`
   * - ``float6_e2m3fnx64``
     - :obj:`tvm_tirx.tirx.script.float6_e2m3fnx64`
   * - ``float6_e3m2fn``
     - :obj:`tvm_tirx.tirx.script.float6_e3m2fn`
   * - ``float6_e3m2fnx2``
     - :obj:`tvm_tirx.tirx.script.float6_e3m2fnx2`
   * - ``float6_e3m2fnx4``
     - :obj:`tvm_tirx.tirx.script.float6_e3m2fnx4`
   * - ``float6_e3m2fnx8``
     - :obj:`tvm_tirx.tirx.script.float6_e3m2fnx8`
   * - ``float6_e3m2fnx16``
     - :obj:`tvm_tirx.tirx.script.float6_e3m2fnx16`
   * - ``float6_e3m2fnx32``
     - :obj:`tvm_tirx.tirx.script.float6_e3m2fnx32`
   * - ``float6_e3m2fnx64``
     - :obj:`tvm_tirx.tirx.script.float6_e3m2fnx64`
   * - ``float4_e2m1fn``
     - :obj:`tvm_tirx.tirx.script.float4_e2m1fn`
   * - ``float4_e2m1fnx2``
     - :obj:`tvm_tirx.tirx.script.float4_e2m1fnx2`
   * - ``float4_e2m1fnx4``
     - :obj:`tvm_tirx.tirx.script.float4_e2m1fnx4`
   * - ``float4_e2m1fnx8``
     - :obj:`tvm_tirx.tirx.script.float4_e2m1fnx8`
   * - ``float4_e2m1fnx16``
     - :obj:`tvm_tirx.tirx.script.float4_e2m1fnx16`
   * - ``float4_e2m1fnx32``
     - :obj:`tvm_tirx.tirx.script.float4_e2m1fnx32`
   * - ``float4_e2m1fnx64``
     - :obj:`tvm_tirx.tirx.script.float4_e2m1fnx64`
   * - ``bfloat16``
     - :obj:`tvm_tirx.tirx.script.bf16`
   * - ``f16``
     - :obj:`tvm_tirx.tirx.script.f16`
   * - ``f32``
     - :obj:`tvm_tirx.tirx.script.f32`
   * - ``f64``
     - :obj:`tvm_tirx.tirx.script.f64`
   * - ``bf16``
     - :obj:`tvm_tirx.tirx.script.bf16`
   * - ``i8``
     - :obj:`tvm_tirx.tirx.script.i8`
   * - ``i16``
     - :obj:`tvm_tirx.tirx.script.i16`
   * - ``i32``
     - :obj:`tvm_tirx.tirx.script.i32`
   * - ``i64``
     - :obj:`tvm_tirx.tirx.script.i64`
   * - ``u8``
     - :obj:`tvm_tirx.tirx.script.u8`
   * - ``u16``
     - :obj:`tvm_tirx.tirx.script.u16`
   * - ``u32``
     - :obj:`tvm_tirx.tirx.script.u32`
   * - ``u64``
     - :obj:`tvm_tirx.tirx.script.u64`
   * - ``boolean``
     - :obj:`tvm_tirx.tirx.script.bool`
   * - ``handle``
     - :obj:`tvm_tirx.tirx.script.handle`
   * - ``void``
     - :obj:`tvm_tirx.tirx.script.void`
   * - ``ptr``
     - :obj:`tvm_tirx.tirx.script.ptr`
   * - ``index_map``
     - :obj:`tvm_tirx.tirx.script.index_map`
   * - ``target``
     - :obj:`tvm_tirx.tirx.script.target`
   * - ``meta_class``
     - :obj:`tvm_tirx.tirx.script.meta_class`
   * - ``bool``
     - :obj:`tvm_tirx.tirx.script.bool`
   * - ``llvm_lookup_intrinsic_id``
     - :obj:`tvm_tirx.codegen.llvm.lookup_intrinsic_id`
   * - ``call_kernel``
     - :obj:`tvm_tirx.tirx.script.call_kernel`
   * - ``cast``
     - :obj:`tvm_tirx.tirx.script.cast`
   * - ``Let``
     - :obj:`tvm_tirx.tirx.script.Let`
   * - ``min``
     - :obj:`tvm_tirx.tirx.script.min`
   * - ``max``
     - :obj:`tvm_tirx.tirx.script.max`
   * - ``register_script_namespace``
     - :obj:`tvm_tirx.tirx.script.register_script_namespace`
   * - ``abs``
     - :obj:`tvm_tirx.tirx.script.abs`
   * - ``acos``
     - :obj:`tvm_tirx.tirx.script.acos`
   * - ``acosh``
     - :obj:`tvm_tirx.tirx.script.acosh`
   * - ``address_of``
     - :obj:`tvm_tirx.tirx.script.address_of`
   * - ``atomic_add``
     - :obj:`tvm_tirx.tirx.script.atomic_add`
   * - ``call_spirv_pure_glsl450``
     - :obj:`tvm_tirx.tirx.script.call_spirv_pure_glsl450`
   * - ``device_scope``
     - :obj:`tvm_tirx.tirx.script.device_scope`
   * - ``prefetch``
     - :obj:`tvm_tirx.tirx.script.prefetch`
   * - ``static_handle``
     - :obj:`tvm_tirx.tirx.script.static_handle`
   * - ``asin``
     - :obj:`tvm_tirx.tirx.script.asin`
   * - ``asinh``
     - :obj:`tvm_tirx.tirx.script.asinh`
   * - ``atan``
     - :obj:`tvm_tirx.tirx.script.atan`
   * - ``atan2``
     - :obj:`tvm_tirx.tirx.script.atan2`
   * - ``atanh``
     - :obj:`tvm_tirx.tirx.script.atanh`
   * - ``bitwise_and``
     - :obj:`tvm_tirx.tirx.script.bitwise_and`
   * - ``bitwise_not``
     - :obj:`tvm_tirx.tirx.script.bitwise_not`
   * - ``bitwise_or``
     - :obj:`tvm_tirx.tirx.script.bitwise_or`
   * - ``bitwise_xor``
     - :obj:`tvm_tirx.tirx.script.bitwise_xor`
   * - ``ceil``
     - :obj:`tvm_tirx.tirx.script.ceil`
   * - ``clz``
     - :obj:`tvm_tirx.tirx.script.clz`
   * - ``copysign``
     - :obj:`tvm_tirx.tirx.script.copysign`
   * - ``cos``
     - :obj:`tvm_tirx.tirx.script.cos`
   * - ``cosh``
     - :obj:`tvm_tirx.tirx.script.cosh`
   * - ``erf``
     - :obj:`tvm_tirx.tirx.script.erf`
   * - ``exp``
     - :obj:`tvm_tirx.tirx.script.exp`
   * - ``exp2``
     - :obj:`tvm_tirx.tirx.script.exp2`
   * - ``exp10``
     - :obj:`tvm_tirx.tirx.script.exp10`
   * - ``gpu_thread_filter``
     - :obj:`tvm_tirx.tirx.script.gpu_thread_filter`
   * - ``gpu_active_thread_selector``
     - :obj:`tvm_tirx.tirx.script.gpu_active_thread_selector`
   * - ``floor``
     - :obj:`tvm_tirx.tirx.script.floor`
   * - ``ceildiv``
     - :obj:`tvm_tirx.tirx.script.ceildiv`
   * - ``floordiv``
     - :obj:`tvm_tirx.tirx.script.floordiv`
   * - ``floormod``
     - :obj:`tvm_tirx.tirx.script.floormod`
   * - ``fmod``
     - :obj:`tvm_tirx.tirx.script.fmod`
   * - ``fma``
     - :obj:`tvm_tirx.tirx.script.fma`
   * - ``hypot``
     - :obj:`tvm_tirx.tirx.script.hypot`
   * - ``if_then_else``
     - :obj:`tvm_tirx.tirx.script.if_then_else`
   * - ``infinity``
     - :obj:`tvm_tirx.tirx.script.infinity`
   * - ``isfinite``
     - :obj:`tvm_tirx.tirx.script.isfinite`
   * - ``isinf``
     - :obj:`tvm_tirx.tirx.script.isinf`
   * - ``isnan``
     - :obj:`tvm_tirx.tirx.script.isnan`
   * - ``isnullptr``
     - :obj:`tvm_tirx.tirx.script.isnullptr`
   * - ``ldexp``
     - :obj:`tvm_tirx.tirx.script.ldexp`
   * - ``likely``
     - :obj:`tvm_tirx.tirx.script.likely`
   * - ``log``
     - :obj:`tvm_tirx.tirx.script.log`
   * - ``log1p``
     - :obj:`tvm_tirx.tirx.script.log1p`
   * - ``log2``
     - :obj:`tvm_tirx.tirx.script.log2`
   * - ``log10``
     - :obj:`tvm_tirx.tirx.script.log10`
   * - ``max_value``
     - :obj:`tvm_tirx.tirx.script.max_value`
   * - ``min_value``
     - :obj:`tvm_tirx.tirx.script.min_value`
   * - ``nearbyint``
     - :obj:`tvm_tirx.tirx.script.nearbyint`
   * - ``nextafter``
     - :obj:`tvm_tirx.tirx.script.nextafter`
   * - ``popcount``
     - :obj:`tvm_tirx.tirx.script.popcount`
   * - ``pow``
     - :obj:`tvm_tirx.tirx.script.pow`
   * - ``round``
     - :obj:`tvm_tirx.tirx.script.round`
   * - ``rsqrt``
     - :obj:`tvm_tirx.tirx.script.rsqrt`
   * - ``shift_left``
     - :obj:`tvm_tirx.tirx.script.shift_left`
   * - ``shift_right``
     - :obj:`tvm_tirx.tirx.script.shift_right`
   * - ``sigmoid``
     - :obj:`tvm_tirx.tirx.script.sigmoid`
   * - ``sin``
     - :obj:`tvm_tirx.tirx.script.sin`
   * - ``sinh``
     - :obj:`tvm_tirx.tirx.script.sinh`
   * - ``sqrt``
     - :obj:`tvm_tirx.tirx.script.sqrt`
   * - ``tan``
     - :obj:`tvm_tirx.tirx.script.tan`
   * - ``tanh``
     - :obj:`tvm_tirx.tirx.script.tanh`
   * - ``gpu_thread_return``
     - :obj:`tvm_tirx.tirx.script.gpu_thread_return`
   * - ``trunc``
     - :obj:`tvm_tirx.tirx.script.trunc`
   * - ``truncdiv``
     - :obj:`tvm_tirx.tirx.script.truncdiv`
   * - ``truncmod``
     - :obj:`tvm_tirx.tirx.script.truncmod`
   * - ``ptr_byte_offset``
     - :obj:`tvm_tirx.tirx.script.ptr_byte_offset`
   * - ``throw_last_error``
     - :obj:`tvm_tirx.tirx.script.throw_last_error`
   * - ``stack_alloca``
     - :obj:`tvm_tirx.tirx.script.stack_alloca`
   * - ``stack_make_shape``
     - :obj:`tvm_tirx.tirx.script.stack_make_shape`
   * - ``stack_make_dltensor``
     - :obj:`tvm_tirx.tirx.script.stack_make_dltensor`
   * - ``call_packed``
     - :obj:`tvm_tirx.tirx.script.call_packed`
   * - ``call_ffi_kernel``
     - :obj:`tvm_tirx.tirx.script.call_ffi_kernel`
   * - ``call_cpacked``
     - :obj:`tvm_tirx.tirx.script.call_cpacked`
   * - ``call_packed_lowered``
     - :obj:`tvm_tirx.tirx.script.call_packed_lowered`
   * - ``call_cpacked_lowered``
     - :obj:`tvm_tirx.tirx.script.call_cpacked_lowered`
   * - ``tensor_data_ptr``
     - :obj:`tvm_tirx.tirx.script.tensor_data_ptr`
   * - ``handle_add_byte_offset``
     - :obj:`tvm_tirx.tirx.script.handle_add_byte_offset`
   * - ``abi_field_set``
     - :obj:`tvm_tirx.tirx.script.abi_field_set`
   * - ``abi_field_get``
     - :obj:`tvm_tirx.tirx.script.abi_field_get`
   * - ``gpu_thread_invariant``
     - :obj:`tvm_tirx.tirx.script.gpu_thread_invariant`
   * - ``gpu_thread_allreduce``
     - :obj:`tvm_tirx.tirx.script.gpu_thread_allreduce`
   * - ``gpu_load_matrix_sync``
     - :obj:`tvm_tirx.tirx.script.gpu_load_matrix_sync`
   * - ``gpu_mma_sync``
     - :obj:`tvm_tirx.tirx.script.gpu_mma_sync`
   * - ``gpu_fill_fragment``
     - :obj:`tvm_tirx.tirx.script.gpu_fill_fragment`
   * - ``gpu_store_matrix_sync``
     - :obj:`tvm_tirx.tirx.script.gpu_store_matrix_sync`
   * - ``gpu_storage_sync``
     - :obj:`tvm_tirx.tirx.script.gpu_storage_sync`
   * - ``cpu_parallel_barrier``
     - :obj:`tvm_tirx.tirx.script.cpu_parallel_barrier`
   * - ``kernel_replace_point``
     - :obj:`tvm_tirx.tirx.script.kernel_replace_point`
   * - ``gpu_warp_shuffle``
     - :obj:`tvm_tirx.tirx.script.gpu_warp_shuffle`
   * - ``gpu_warp_shuffle_up``
     - :obj:`tvm_tirx.tirx.script.gpu_warp_shuffle_up`
   * - ``gpu_warp_shuffle_down``
     - :obj:`tvm_tirx.tirx.script.gpu_warp_shuffle_down`
   * - ``gpu_warp_shuffle_xor``
     - :obj:`tvm_tirx.tirx.script.gpu_warp_shuffle_xor`
   * - ``gpu_warp_activemask``
     - :obj:`tvm_tirx.tirx.script.gpu_warp_activemask`
   * - ``assume``
     - :obj:`tvm_tirx.tirx.script.assume`
   * - ``assume_aligned``
     - :obj:`tvm_tirx.tirx.script.assume_aligned`
   * - ``undef``
     - :obj:`tvm_tirx.tirx.script.undef`
   * - ``alloc_workspace``
     - :obj:`tvm_tirx.tirx.script.alloc_workspace`
   * - ``free_workspace``
     - :obj:`tvm_tirx.tirx.script.free_workspace`
   * - ``vscale``
     - :obj:`tvm_tirx.tirx.script.vscale`
   * - ``ignore_loop_partition``
     - :obj:`tvm_tirx.tirx.script.ignore_loop_partition`
   * - ``reinterpret``
     - :obj:`tvm_tirx.tirx.script.reinterpret`
   * - ``call_extern``
     - :obj:`tvm_tirx.tirx.script.call_extern`
   * - ``call_intrin``
     - :obj:`tvm_tirx.tirx.script.call_intrin`
   * - ``call_llvm_intrin``
     - :obj:`tvm_tirx.tirx.script.call_llvm_intrin`
   * - ``call_llvm_pure_intrin``
     - :obj:`tvm_tirx.tirx.script.call_llvm_pure_intrin`
   * - ``call_pure_extern``
     - :obj:`tvm_tirx.tirx.script.call_pure_extern`
   * - ``vector_low``
     - :obj:`tvm_tirx.tirx.script.vector_low`
   * - ``vector_high``
     - :obj:`tvm_tirx.tirx.script.vector_high`
   * - ``vector_combine``
     - :obj:`tvm_tirx.tirx.script.vector_combine`
   * - ``get_active_lane_mask``
     - :obj:`tvm_tirx.tirx.script.get_active_lane_mask`
   * - ``masked_load``
     - :obj:`tvm_tirx.tirx.script.masked_load`
   * - ``masked_store``
     - :obj:`tvm_tirx.tirx.script.masked_store`
   * - ``gpu_dp4a``
     - :obj:`tvm_tirx.tirx.script.gpu_dp4a`
   * - ``fabs``
     - :obj:`tvm_tirx.tirx.script.fabs`
   * - ``logical_and``
     - :obj:`tvm_tirx.tirx.script.logical_and`
   * - ``logical_or``
     - :obj:`tvm_tirx.tirx.script.logical_or`
   * - ``logical_not``
     - :obj:`tvm_tirx.tirx.script.logical_not`
   * - ``select``
     - :obj:`tvm_tirx.tirx.script.select`
   * - ``if_then_else_``
     - :obj:`tvm_tirx.tirx.script.if_then_else_`
   * - ``and_``
     - :obj:`tvm_tirx.tirx.script.and_`
   * - ``or_``
     - :obj:`tvm_tirx.tirx.script.or_`
   * - ``not_``
     - :obj:`tvm_tirx.tirx.script.not_`
   * - ``lt_``
     - :obj:`tvm_tirx.tirx.script.lt_`
   * - ``le_``
     - :obj:`tvm_tirx.tirx.script.le_`
   * - ``gt_``
     - :obj:`tvm_tirx.tirx.script.gt_`
   * - ``ge_``
     - :obj:`tvm_tirx.tirx.script.ge_`
   * - ``eq_``
     - :obj:`tvm_tirx.tirx.script.eq_`
   * - ``ne_``
     - :obj:`tvm_tirx.tirx.script.ne_`
   * - ``add_to_parent``
     - :obj:`tvm_tirx.tirx.script.add_to_parent`
   * - ``assert_``
     - :obj:`tvm_tirx.tirx.script.assert_`
   * - ``break_``
     - :obj:`tvm_tirx.tirx.script.break_`
   * - ``continue_``
     - :obj:`tvm_tirx.tirx.script.continue_`
   * - ``else_``
     - :obj:`tvm_tirx.tirx.script.else_`
   * - ``evaluate``
     - :obj:`tvm_tirx.tirx.script.evaluate`
   * - ``grid``
     - :obj:`tvm_tirx.tirx.script.grid`
   * - ``if_``
     - :obj:`tvm_tirx.tirx.script.if_`
   * - ``region``
     - :obj:`tvm_tirx.tirx.script.region`
   * - ``return_``
     - :obj:`tvm_tirx.tirx.script.return_`
   * - ``then_``
     - :obj:`tvm_tirx.tirx.script.then_`
   * - ``unpack_``
     - :obj:`tvm_tirx.tirx.script.unpack_`
   * - ``while_``
     - :obj:`tvm_tirx.tirx.script.while_`

.. autosummary::
   :toctree: generated/

   function

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``function_``
     - :obj:`tvm_tirx.tirx.script.function_`
   * - ``arg_``
     - :obj:`tvm_tirx.tirx.script.arg_`
   * - ``func_name_``
     - :obj:`tvm_tirx.tirx.script.func_name_`
   * - ``func_ret_type_``
     - :obj:`tvm_tirx.tirx.script.func_ret`
   * - ``func_attr``
     - :obj:`tvm_tirx.tirx.script.func_attr`
   * - ``call_global_var_``
     - :obj:`tvm_tirx.tirx.script.call_global_var_`
   * - ``bind``
     - :obj:`tvm_tirx.tirx.script.bind`
   * - ``bind_``
     - :obj:`tvm_tirx.tirx.script.bind_`
   * - ``decl_mutable_cell_``
     - :obj:`tvm_tirx.tirx.script.decl_mutable_cell_`
   * - ``set_mutable_cell_``
     - :obj:`tvm_tirx.tirx.script.set_mutable_cell_`
   * - ``emit_``
     - :obj:`tvm_tirx.tirx.script.emit`
   * - ``setitem_``
     - :obj:`tvm_tirx.tirx.script.setitem_`
   * - ``setattr_``
     - :obj:`tvm_tirx.tirx.script.setattr_`
   * - ``tensor_store``
     - :obj:`tvm_tirx.tirx.script.tensor_store`
   * - ``for_``
     - :obj:`tvm_tirx.tirx.script.for_`
   * - ``range_``
     - :obj:`tvm_tirx.tirx.script.range_`
   * - ``serial``
     - :obj:`tvm_tirx.tirx.script.serial`
   * - ``parallel``
     - :obj:`tvm_tirx.tirx.script.parallel`
   * - ``vectorized``
     - :obj:`tvm_tirx.tirx.script.vectorized`
   * - ``unroll``
     - :obj:`tvm_tirx.tirx.script.unroll`
   * - ``device_entry``
     - :obj:`tvm_tirx.tirx.script.device_entry`
   * - ``device_context``
     - :obj:`tvm_tirx.tirx.script.device_context`
   * - ``compute_scope``
     - :obj:`tvm_tirx.tirx.script.compute_scope`
   * - ``parallel_launch``
     - :obj:`tvm_tirx.tirx.script.parallel_launch`
   * - ``launch_thread``
     - :obj:`tvm_tirx.tirx.script.launch_thread`
   * - ``func_ret``
     - :obj:`tvm_tirx.tirx.script.func_ret`
   * - ``emit``
     - :obj:`tvm_tirx.tirx.script.emit`
   * - ``frame_scope``
     - :obj:`tvm_tirx.tirx.script.frame_scope`
   * - ``tensor_indices``
     - :obj:`tvm_tirx.tirx.script.tensor_indices`

Constants and construction helpers
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``MISSING``
     - :obj:`tvm_tirx.script.ir_builder.MISSING`
   * - ``R``
     - :obj:`tvm_tirx.tirx.layout.R`
   * - ``S``
     - :obj:`tvm_tirx.tirx.layout.S`
   * - ``let``
     - :obj:`tvm_tirx.tirx.script.let`
   * - ``supports_mutable_declarations``
     - :obj:`tvm_tirx.tirx.script.supports_mutable_declarations`

Namespaces
----------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.tirx.script.ir_builder.frame <tirx.script.ir_builder.frame>`
     - IRBuilder for TIR
   * - :doc:`tvm_tirx.tirx.script.ir_builder.external_kernel <tirx.script.ir_builder.external_kernel>`
     - External kernel integration fro TIR
   * - :doc:`tvm_tirx.tirx.script.ir_builder.parser_protocol <tirx.script.ir_builder.parser_protocol>`
     - TIRx implementation of the shared source-to-builder protocol.
   * - :doc:`tvm_tirx.tirx.script.ir_builder.triton <tirx.script.ir_builder.triton>`
     - Triton kernel integration with TIR

.. toctree::
   :hidden:
   :maxdepth: 2

   tirx.script.ir_builder.frame
   tirx.script.ir_builder.external_kernel
   tirx.script.ir_builder.parser_protocol
   tirx.script.ir_builder.triton

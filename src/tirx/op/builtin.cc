/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tirx/op/builtin.cc
 *
 *  builtin intrinsic operators.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/tirx/attrs.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {
namespace builtin {

TVM_FFI_STATIC_INIT_BLOCK() {
  TensorMapEncodeTiledAttr::RegisterReflection();
  ffi::reflection::GlobalDef().def(
      "tirx.TensorMapEncodeTiledAttr",
      [](DLDataType descriptor_dtype, int64_t rank, int64_t interleave, int64_t swizzle,
         int64_t l2_promotion, int64_t oob_fill, int64_t force_cu_dtype) {
        auto attrs = ffi::make_object<TensorMapEncodeTiledAttr>();
        attrs->descriptor_dtype = descriptor_dtype;
        attrs->rank = rank;
        attrs->interleave = interleave;
        attrs->swizzle = swizzle;
        attrs->l2_promotion = l2_promotion;
        attrs->oob_fill = oob_fill;
        attrs->force_cu_dtype = force_cu_dtype;
        return Attrs(attrs);
      });
  CallFFIKernelAttr::RegisterReflection();
  ffi::reflection::GlobalDef().def("tirx.CallFFIKernelAttr",
                                   [](ffi::Array<ffi::String> launch_params) {
                                     auto attrs = ffi::make_object<CallFFIKernelAttr>();
                                     attrs->launch_params = std::move(launch_params);
                                     return Attrs(attrs);
                                   });

  // Script metadata extends the canonical primitive operators registered by IR.

  OpDef("prim.likely")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("likely"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.if_then_else")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("if_then_else"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.vscale")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("vscale"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.ceil")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("ceil"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.log2")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("log2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.clz")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("clz"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
}

const Op& reinterpret() {
  static const Op op = Op::Get("tirx.reinterpret");
  return op;
}

const Op& thread_return() {
  static const Op op = Op::Get("tirx.thread_return");
  return op;
}

const Op& filter() {
  static const Op op = Op::Get("tirx.filter");
  return op;
}

const Op& selector() {
  static const Op op = Op::Get("tirx.selector");
  return op;
}

const Op& address_of() {
  static const Op op = Op::Get("tirx.address_of");
  return op;
}

const Op& q_multiply_shift() {
  static const Op op = Op::Get("tirx.q_multiply_shift");
  return op;
}

const Op& q_multiply_shift_per_axis() {
  static const Op op = Op::Get("tirx.q_multiply_shift_per_axis");
  return op;
}

const Op& isnullptr() {
  static const Op op = Op::Get("tirx.isnullptr");
  return op;
}

const Op& isnan() {
  static const Op op = Op::Get("tirx.isnan");
  return op;
}

const Op& popcount() {
  static const Op op = Op::Get("tirx.popcount");
  return op;
}

const Op& fma() {
  static const Op op = Op::Get("tirx.fma");
  return op;
}

const Op& call_extern() {
  static const Op op = Op::Get("tirx.call_extern");
  return op;
}

const Op& call_pure_extern() {
  static const Op op = Op::Get("tirx.call_pure_extern");
  return op;
}

const Op& call_llvm_intrin() {
  static const Op op = Op::Get("tirx.call_llvm_intrin");
  return op;
}

const Op& call_llvm_pure_intrin() {
  static const Op op = Op::Get("tirx.call_llvm_pure_intrin");
  return op;
}

const Op& call_spirv_pure_glsl450() {
  static const Op op = Op::Get("tirx.call_spirv_pure_glsl450");
  return op;
}

const Op& prefetch() {
  static const Op op = Op::Get("tirx.prefetch");
  return op;
}

const Op& tvm_access_ptr() {
  static const Op op = Op::Get("tirx.tvm_access_ptr");
  return op;
}

const Op& ptr_byte_offset() {
  static const Op op = Op::Get("tirx.ptr_byte_offset");
  return op;
}

const Op& tvm_static_handle() {
  static const Op op = Op::Get("tirx.tvm_static_handle");
  return op;
}

const Op& handle_add_byte_offset() {
  static const Op op = Op::Get("tirx.handle_add_byte_offset");
  return op;
}

const Op& tvm_struct_get() {
  static const Op op = Op::Get("tirx.tvm_struct_get");
  return op;
}

const Op& tvm_struct_set() {
  static const Op op = Op::Get("tirx.tvm_struct_set");
  return op;
}

const Op& tvm_throw_last_error() {
  static const Op op = Op::Get("tirx.tvm_throw_last_error");
  return op;
}

const Op& tvm_stack_alloca() {
  static const Op op = Op::Get("tirx.tvm_stack_alloca");
  return op;
}

const Op& tvm_stack_make_shape() {
  static const Op op = Op::Get("tirx.tvm_stack_make_shape");
  return op;
}

const Op& tvm_stack_make_array() {
  static const Op op = Op::Get("tirx.tvm_stack_make_array");
  return op;
}

const Op& tvm_call_packed() {
  static const Op op = Op::Get("tirx.tvm_call_packed");
  return op;
}

const Op& tensormap_encode_tiled() {
  static const Op op = Op::Get("tirx.tensormap_encode_tiled");
  return op;
}

const Op& call_ffi_kernel() {
  static const Op op = Op::Get("tirx.call_ffi_kernel");
  return op;
}

const Op& tvm_call_cpacked() {
  static const Op op = Op::Get("tirx.tvm_call_cpacked");
  return op;
}

const Op& tvm_thread_invariant() {
  static const Op op = Op::Get("tirx.tvm_thread_invariant");
  return op;
}

const Op& tvm_call_packed_lowered() {
  static const Op op = Op::Get("tirx.tvm_call_packed_lowered");
  return op;
}

const Op& tvm_call_cpacked_lowered() {
  static const Op op = Op::Get("tirx.tvm_call_cpacked_lowered");
  return op;
}

const Op& tvm_storage_sync() {
  static const Op op = Op::Get("tirx.tvm_storage_sync");
  return op;
}

const Op& tvm_kernel_replace_point() {
  static const Op op = Op::Get("tirx.tvm_kernel_replace_point");
  return op;
}

const Op& tvm_warp_shuffle() {
  static const Op op = Op::Get("tirx.tvm_warp_shuffle");
  return op;
}

const Op& tvm_warp_shuffle_up() {
  static const Op op = Op::Get("tirx.tvm_warp_shuffle_up");
  return op;
}

const Op& tvm_warp_shuffle_down() {
  static const Op op = Op::Get("tirx.tvm_warp_shuffle_down");
  return op;
}

const Op& tvm_warp_shuffle_xor() {
  static const Op op = Op::Get("tirx.tvm_warp_shuffle_xor");
  return op;
}

const Op& tvm_warp_activemask() {
  static const Op op = Op::Get("tirx.tvm_warp_activemask");
  return op;
}

const Op& tvm_thread_allreduce() {
  static const Op op = Op::Get("tirx.tvm_thread_allreduce");
  return op;
}

const Op& cooperative_tensor_fill() {
  static const Op op = Op::Get("tirx.cooperative_tensor_fill");
  return op;
}

const Op& cooperative_tensor_load() {
  static const Op op = Op::Get("tirx.cooperative_tensor_load");
  return op;
}

const Op& cooperative_tensor_store() {
  static const Op op = Op::Get("tirx.cooperative_tensor_store");
  return op;
}

const Op& cooperative_tensor_multiply_accumulate() {
  static const Op op = Op::Get("tirx.cooperative_tensor_multiply_accumulate");
  return op;
}

const Op& vectorhigh() {
  static const Op op = Op::Get("tirx.vectorhigh");
  return op;
}

const Op& vectorlow() {
  static const Op op = Op::Get("tirx.vectorlow");
  return op;
}

const Op& vectorcombine() {
  static const Op op = Op::Get("tirx.vectorcombine");
  return op;
}

const Op& dp4a() {
  static const Op op = Op::Get("tirx.dp4a");
  return op;
}

const Op& atomic_add() {
  static const Op op = Op::Get("tirx.atomic_add");
  return op;
}

const Op& nd_mem_alloc_with_scope() {
  static const Op op = Op::Get("tirx.nd_mem_alloc_with_scope");
  return op;
}

const Op& texture2d_store() {
  static const Op op = Op::Get("tirx.texture2d_store");
  return op;
}

const Op& texture2d_load() {
  static const Op op = Op::Get("tirx.texture2d_load");
  return op;
}

const Op& assume() {
  static const Op op = Op::Get("tirx.assume");
  return op;
}

const Op& undef() {
  static const Op op = Op::Get("tirx.undef");
  return op;
}

const Op& get_active_lane_mask() {
  static const Op op = Op::Get("tirx.get_active_lane_mask");
  return op;
}

const Op& masked_load() {
  static const Op op = Op::Get("tirx.masked_load");
  return op;
}

const Op& masked_store() {
  static const Op op = Op::Get("tirx.masked_store");
  return op;
}

const Op& ignore_loop_partition() {
  static const Op op = Op::Get("tirx.ignore_loop_partition");
  return op;
}

const Op& buffer_offset() {
  static const Op op = Op::Get("tirx.buffer_offset");
  return op;
}

const Op& buffer_data() {
  static const Op op = Op::Get("tirx.buffer_data");
  return op;
}

const Op& print_buffer() {
  static const Op op = Op::Get("tirx.print_buffer");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.reinterpret")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("reinterpret"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst))
      .set_num_inputs(1);

  OpDef("tirx.thread_return")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("thread_return"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kControlJump))
      .set_num_inputs(0);

  // tirx.filter: escape hatch for non-canonical thread-set filter predicates
  // used as an IfThenElse condition. (var, cond) -- ``var`` names the
  // active-set axis the compiler should collapse to a singleton if it cannot
  // statically analyze ``cond``. Canonical predicates (see
  // ``analysis/filter_canonical.h``) should appear bare in ``if`` conditions
  // without this wrapper.

  OpDef("tirx.filter")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("filter"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(2)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.selector")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("selector"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(2)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.address_of")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("address_of"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_num_inputs(1);

  OpDef("tirx.q_multiply_shift")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("q_multiply_shift"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(3)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.q_multiply_shift_per_axis")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("q_multiply_shift_per_axis"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(7)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.isnullptr")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("isnullptr"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(1)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.isnan")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("isnan"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(1)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.popcount")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("popcount"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(1)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.fma")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("fma"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(3)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.call_extern")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_extern"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.call_pure_extern")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_pure_extern"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.call_llvm_intrin")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_llvm_intrin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.call_llvm_pure_intrin")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_llvm_pure_intrin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.call_spirv_pure_glsl450")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_spirv_pure_glsl450"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.prefetch")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("prefetch"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_access_ptr")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_access_ptr"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(5)
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kSpecialCallArg));

  OpDef("tirx.ptr_byte_offset")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("ptr_byte_offset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(3)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.tvm_static_handle")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_static_handle"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(0)
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kSpecialCallArg));

  OpDef("tirx.handle_add_byte_offset")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("handle_add_byte_offset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(2)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.tvm_struct_get")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_struct_get"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(3)
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kLast));

  OpDef("tirx.tvm_struct_set")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_struct_set"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(4)
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kUpdateState));

  OpDef("tirx.tvm_throw_last_error")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_throw_last_error"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(0)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_stack_alloca")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_stack_alloca"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(2)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_stack_make_shape")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_stack_make_shape"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_stack_make_array")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_stack_make_array"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(6)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  // When num_inputs are not set, the function is assumed to be variable length.

  OpDef("tirx.tvm_call_packed")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_packed"));

  OpDef("tirx.tensormap_encode_tiled")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tensormap_encode_tiled"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.call_ffi_kernel")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_ffi_kernel"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_call_cpacked")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_cpacked"));

  OpDef("tirx.tvm_thread_invariant")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_thread_invariant"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(1)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.tvm_call_packed_lowered")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_packed_lowered"));

  OpDef("tirx.tvm_call_cpacked_lowered")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_cpacked_lowered"));

  // TODO(tvm-team) revisit storage sync once we have a good memory hierachy structure.

  OpDef("tirx.tvm_storage_sync")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_storage_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_kernel_replace_point")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_kernel_replace_point"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(0)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_warp_shuffle")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_warp_shuffle"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_warp_shuffle_up")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_warp_shuffle_up"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_warp_shuffle_down")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_warp_shuffle_down"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_warp_shuffle_xor")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_warp_shuffle_xor"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_warp_activemask")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_warp_activemask"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_thread_allreduce")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_thread_allreduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cooperative_tensor_fill")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cooperative_tensor_fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cooperative_tensor_load")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cooperative_tensor_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cooperative_tensor_store")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cooperative_tensor_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cooperative_tensor_multiply_accumulate")
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("cooperative_tensor_multiply_accumulate"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.vectorhigh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("vectorhigh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.vectorlow")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("vectorlow"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.vectorcombine")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("vectorcombine"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.dp4a")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("dp4a"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.atomic_add")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("atomic_add"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.nd_mem_alloc_with_scope")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("nd_mem_alloc_with_scope"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.texture2d_store")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("texture2d_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TVectorizable>("TVectorizable", true)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.texture2d_load")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("texture2d_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TVectorizable>("TVectorizable", true)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.assume")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("assume"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kEmbedInfo))
      .set_num_inputs(1);

  OpDef("tirx.undef")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("undef"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState))
      .set_num_inputs(0);

  OpDef("tirx.get_active_lane_mask")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("get_active_lane_mask"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(2)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.masked_load")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("masked_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.masked_store")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("masked_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kUpdateState));

  OpDef("tirx.ignore_loop_partition")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("ignore_loop_partition"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(1)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kNone));

  OpDef("tirx.buffer_offset")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("buffer_offset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(2)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.buffer_data")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("buffer_data"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_num_inputs(1)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.print_buffer")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("print_buffer"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm

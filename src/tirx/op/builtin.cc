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

#define TVM_DEFINE_CACHED_OP_GETTER(Name, RegisteredName) \
  const Op& Name() {                                      \
    static const Op op = Op::Get(RegisteredName);         \
    return op;                                            \
  }

TVM_DEFINE_CACHED_OP_GETTER(reinterpret, "tirx.reinterpret")
TVM_DEFINE_CACHED_OP_GETTER(thread_return, "tirx.thread_return")
TVM_DEFINE_CACHED_OP_GETTER(filter, "tirx.filter")
TVM_DEFINE_CACHED_OP_GETTER(selector, "tirx.selector")
TVM_DEFINE_CACHED_OP_GETTER(address_of, "tirx.address_of")
TVM_DEFINE_CACHED_OP_GETTER(q_multiply_shift, "tirx.q_multiply_shift")
TVM_DEFINE_CACHED_OP_GETTER(q_multiply_shift_per_axis, "tirx.q_multiply_shift_per_axis")
TVM_DEFINE_CACHED_OP_GETTER(isnullptr, "tirx.isnullptr")
TVM_DEFINE_CACHED_OP_GETTER(isnan, "tirx.isnan")
TVM_DEFINE_CACHED_OP_GETTER(popcount, "tirx.popcount")
TVM_DEFINE_CACHED_OP_GETTER(fma, "tirx.fma")
TVM_DEFINE_CACHED_OP_GETTER(call_extern, "tirx.call_extern")
TVM_DEFINE_CACHED_OP_GETTER(call_pure_extern, "tirx.call_pure_extern")
TVM_DEFINE_CACHED_OP_GETTER(call_llvm_intrin, "tirx.call_llvm_intrin")
TVM_DEFINE_CACHED_OP_GETTER(call_llvm_pure_intrin, "tirx.call_llvm_pure_intrin")
TVM_DEFINE_CACHED_OP_GETTER(call_spirv_pure_glsl450, "tirx.call_spirv_pure_glsl450")
TVM_DEFINE_CACHED_OP_GETTER(prefetch, "tirx.prefetch")
TVM_DEFINE_CACHED_OP_GETTER(tvm_access_ptr, "tirx.tvm_access_ptr")
TVM_DEFINE_CACHED_OP_GETTER(ptr_byte_offset, "tirx.ptr_byte_offset")
TVM_DEFINE_CACHED_OP_GETTER(tvm_static_handle, "tirx.tvm_static_handle")
TVM_DEFINE_CACHED_OP_GETTER(handle_add_byte_offset, "tirx.handle_add_byte_offset")
TVM_DEFINE_CACHED_OP_GETTER(tvm_struct_get, "tirx.tvm_struct_get")
TVM_DEFINE_CACHED_OP_GETTER(tvm_struct_set, "tirx.tvm_struct_set")
TVM_DEFINE_CACHED_OP_GETTER(tvm_throw_last_error, "tirx.tvm_throw_last_error")
TVM_DEFINE_CACHED_OP_GETTER(tvm_stack_alloca, "tirx.tvm_stack_alloca")
TVM_DEFINE_CACHED_OP_GETTER(tvm_stack_make_shape, "tirx.tvm_stack_make_shape")
TVM_DEFINE_CACHED_OP_GETTER(tvm_stack_make_array, "tirx.tvm_stack_make_array")
TVM_DEFINE_CACHED_OP_GETTER(tvm_call_packed, "tirx.tvm_call_packed")
TVM_DEFINE_CACHED_OP_GETTER(tensormap_encode_tiled, "tirx.tensormap_encode_tiled")
TVM_DEFINE_CACHED_OP_GETTER(call_ffi_kernel, "tirx.call_ffi_kernel")
TVM_DEFINE_CACHED_OP_GETTER(tvm_call_cpacked, "tirx.tvm_call_cpacked")
TVM_DEFINE_CACHED_OP_GETTER(tvm_thread_invariant, "tirx.tvm_thread_invariant")
TVM_DEFINE_CACHED_OP_GETTER(tvm_call_packed_lowered, "tirx.tvm_call_packed_lowered")
TVM_DEFINE_CACHED_OP_GETTER(tvm_call_cpacked_lowered, "tirx.tvm_call_cpacked_lowered")
TVM_DEFINE_CACHED_OP_GETTER(tvm_storage_sync, "tirx.tvm_storage_sync")
TVM_DEFINE_CACHED_OP_GETTER(tvm_kernel_replace_point, "tirx.tvm_kernel_replace_point")
TVM_DEFINE_CACHED_OP_GETTER(tvm_warp_shuffle, "tirx.tvm_warp_shuffle")
TVM_DEFINE_CACHED_OP_GETTER(tvm_warp_shuffle_up, "tirx.tvm_warp_shuffle_up")
TVM_DEFINE_CACHED_OP_GETTER(tvm_warp_shuffle_down, "tirx.tvm_warp_shuffle_down")
TVM_DEFINE_CACHED_OP_GETTER(tvm_warp_shuffle_xor, "tirx.tvm_warp_shuffle_xor")
TVM_DEFINE_CACHED_OP_GETTER(tvm_warp_activemask, "tirx.tvm_warp_activemask")
TVM_DEFINE_CACHED_OP_GETTER(tvm_thread_allreduce, "tirx.tvm_thread_allreduce")
TVM_DEFINE_CACHED_OP_GETTER(cooperative_tensor_fill, "tirx.cooperative_tensor_fill")
TVM_DEFINE_CACHED_OP_GETTER(cooperative_tensor_load, "tirx.cooperative_tensor_load")
TVM_DEFINE_CACHED_OP_GETTER(cooperative_tensor_store, "tirx.cooperative_tensor_store")
TVM_DEFINE_CACHED_OP_GETTER(cooperative_tensor_multiply_accumulate,
                            "tirx.cooperative_tensor_multiply_accumulate")
TVM_DEFINE_CACHED_OP_GETTER(vectorhigh, "tirx.vectorhigh")
TVM_DEFINE_CACHED_OP_GETTER(vectorlow, "tirx.vectorlow")
TVM_DEFINE_CACHED_OP_GETTER(vectorcombine, "tirx.vectorcombine")
TVM_DEFINE_CACHED_OP_GETTER(dp4a, "tirx.dp4a")
TVM_DEFINE_CACHED_OP_GETTER(atomic_add, "tirx.atomic_add")
TVM_DEFINE_CACHED_OP_GETTER(nd_mem_alloc_with_scope, "tirx.nd_mem_alloc_with_scope")
TVM_DEFINE_CACHED_OP_GETTER(texture2d_store, "tirx.texture2d_store")
TVM_DEFINE_CACHED_OP_GETTER(texture2d_load, "tirx.texture2d_load")
TVM_DEFINE_CACHED_OP_GETTER(assume, "tirx.assume")
TVM_DEFINE_CACHED_OP_GETTER(undef, "tirx.undef")
TVM_DEFINE_CACHED_OP_GETTER(get_active_lane_mask, "tirx.get_active_lane_mask")
TVM_DEFINE_CACHED_OP_GETTER(masked_load, "tirx.masked_load")
TVM_DEFINE_CACHED_OP_GETTER(masked_store, "tirx.masked_store")
TVM_DEFINE_CACHED_OP_GETTER(ignore_loop_partition, "tirx.ignore_loop_partition")
TVM_DEFINE_CACHED_OP_GETTER(buffer_offset, "tirx.buffer_offset")
TVM_DEFINE_CACHED_OP_GETTER(buffer_data, "tirx.buffer_data")
TVM_DEFINE_CACHED_OP_GETTER(print_buffer, "tirx.print_buffer")

#undef TVM_DEFINE_CACHED_OP_GETTER

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.reinterpret")
      .add_arg("x", "The input value.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("reinterpret"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.thread_return")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("thread_return"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kControlJump));

  // tirx.filter: escape hatch for non-canonical thread-set filter predicates
  // used as an IfThenElse condition. (var, cond) -- ``var`` names the
  // active-set axis the compiler should collapse to a singleton if it cannot
  // statically analyze ``cond``. Canonical predicates (see
  // ``analysis/filter_canonical.h``) should appear bare in ``if`` conditions
  // without this wrapper.

  OpDef("tirx.filter")
      .add_arg("var", "The thread-axis variable.")
      .add_arg("pred", "The predicate.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("filter"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.selector")
      .add_arg("var", "The thread-axis variable.")
      .add_arg("pred", "The predicate.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("selector"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.address_of")
      .add_arg("obj", "The referenced object.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("address_of"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.q_multiply_shift")
      .add_arg("x", "The input value.")
      .add_arg("y", "The second input value.")
      .add_arg("q", "The number of fractional bits.")
      .add_arg("s", "The right shift amount.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("q_multiply_shift"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.q_multiply_shift_per_axis")
      .add_arg("x", "The input value.")
      .add_arg("y", "The second input value.")
      .add_arg("ls", "The left shift amount.")
      .add_arg("rs", "The right shift amount.")
      .add_arg("q", "The number of fractional bits.")
      .add_arg("is_lshift_required", "Whether the left shift is required.")
      .add_arg("is_rshift_required", "Whether the right shift is required.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("q_multiply_shift_per_axis"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.isnullptr")
      .add_arg("x", "The input value.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("isnullptr"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.isnan")
      .add_arg("x", "The input value.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("isnan"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.popcount")
      .add_arg("x", "The input value.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("popcount"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.fma")
      .add_arg("x", "The input value.")
      .add_arg("y", "The second input value.")
      .add_arg("z", "The third input value.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("fma"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.call_extern")
      .add_arg("func_name", "The function name.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_extern"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.call_pure_extern")
      .add_arg("func_name", "The function name.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_pure_extern"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.call_llvm_intrin")
      .add_arg("intrin_id", "The intrinsic identifier.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_llvm_intrin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.call_llvm_pure_intrin")
      .add_arg("intrin_id", "The intrinsic identifier.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_llvm_pure_intrin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.call_spirv_pure_glsl450")
      .add_arg("intrin_id", "The intrinsic identifier.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_spirv_pure_glsl450"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.prefetch")
      .add_arg("ptr", "The pointer.")
      .add_arg("rw", "The read/write mode.")
      .add_arg("locality", "The locality hint.")
      .add_arg("cache_type", "The cache policy.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("prefetch"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_access_ptr")
      .add_arg("ptype", "The pointer type.")
      .add_arg("data", "The input data.")
      .add_arg("offset", "The offset.")
      .add_arg("extent", "The extent.")
      .add_arg("rw_mask", "The read/write mask.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_access_ptr"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kSpecialCallArg));

  OpDef("tirx.ptr_byte_offset")
      .add_arg("data", "Base pointer.")
      .add_arg("byte_offset", "Offset in bytes.")
      .add_arg("dtype", "Type annotation for pointed-to elements.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("ptr_byte_offset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.tvm_static_handle")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_static_handle"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kSpecialCallArg));

  OpDef("tirx.handle_add_byte_offset")
      .add_arg("handle", "The handle.")
      .add_arg("offset", "The offset.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("handle_add_byte_offset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.tvm_struct_get")
      .add_arg("arr", "The array.")
      .add_arg("index", "The index.")
      .add_arg("field", "The field index.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_struct_get"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kLast));

  OpDef("tirx.tvm_struct_set")
      .add_arg("arr", "The array.")
      .add_arg("index", "The index.")
      .add_arg("field", "The field index.")
      .add_arg("value", "The value to use.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_struct_set"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kUpdateState));

  OpDef("tirx.tvm_throw_last_error")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_throw_last_error"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_stack_alloca")
      .add_arg("dtype_str", "The data type name.")
      .add_arg("num", "The number of entries.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_stack_alloca"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_stack_make_shape")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_stack_make_shape"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_stack_make_array")
      .add_arg("data", "The input data.")
      .add_arg("shape", "The shape.")
      .add_arg("strides", "The strides.")
      .add_arg("ndim", "The number of dimensions.")
      .add_arg("arr_dtype", "The array data type.")
      .add_arg("elem_offset", "The element offset.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_stack_make_array"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_call_packed")
      .add_arg("func_name", "The function name.")
      .allow_extra_args()
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_packed"));

  OpDef("tirx.tensormap_encode_tiled")
      .add_arg("descriptor", "The descriptor.")
      .add_arg("data", "The input data.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tensormap_encode_tiled"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.call_ffi_kernel")
      .add_arg("kernel", "The kernel.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_ffi_kernel"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_call_cpacked")
      .add_arg("func_name", "The function name.")
      .allow_extra_args()
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_cpacked"));

  OpDef("tirx.tvm_thread_invariant")
      .add_arg("cond", "The condition.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_thread_invariant"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.tvm_call_packed_lowered")
      .add_arg("func_name", "The function name.")
      .add_arg("args_stack", "The argument stack.")
      .add_arg("begin", "The start index.")
      .add_arg("end", "The end index.")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_packed_lowered"));

  OpDef("tirx.tvm_call_cpacked_lowered")
      .add_arg("func_name", "The function name.")
      .add_arg("args_stack", "The argument stack.")
      .add_arg("begin", "The start index.")
      .add_arg("end", "The end index.")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_cpacked_lowered"));

  // TODO(tvm-team) revisit storage sync once we have a good memory hierachy structure.

  OpDef("tirx.tvm_storage_sync")
      .add_arg("storage_scope", "The storage scope.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_storage_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_kernel_replace_point")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_kernel_replace_point"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_warp_shuffle")
      .add_arg("mask", "The mask.")
      .add_arg("value", "The value to use.")
      .add_arg("warp_id", "The warp identifier.")
      .add_arg("width", "The width.")
      .add_arg("warp_size", "The number of threads per warp.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_warp_shuffle"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_warp_shuffle_up")
      .add_arg("mask", "The mask.")
      .add_arg("value", "The value to use.")
      .add_arg("offset", "The offset.")
      .add_arg("width", "The width.")
      .add_arg("warp_size", "The number of threads per warp.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_warp_shuffle_up"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_warp_shuffle_down")
      .add_arg("mask", "The mask.")
      .add_arg("value", "The value to use.")
      .add_arg("offset", "The offset.")
      .add_arg("width", "The width.")
      .add_arg("warp_size", "The number of threads per warp.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_warp_shuffle_down"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_warp_shuffle_xor")
      .add_arg("mask", "The mask.")
      .add_arg("value", "The value to use.")
      .add_arg("lane_mask", "The lane mask.")
      .add_arg("width", "The width.")
      .add_arg("warp_size", "The number of threads per warp.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_warp_shuffle_xor"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_warp_activemask")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_warp_activemask"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_thread_allreduce")
      .add_arg("size", "The size.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_thread_allreduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cooperative_tensor_fill")
      .add_arg("d", "The D operand.")
      .add_arg("index", "The index.")
      .add_arg("value", "The value to use.")
      .add_arg("rows", "The number of rows.")
      .add_arg("cols", "The number of columns.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cooperative_tensor_fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cooperative_tensor_load")
      .add_arg("d", "The D operand.")
      .add_arg("index", "The index.")
      .add_arg("ptr", "The pointer.")
      .add_arg("stride", "The stride.")
      .add_arg("rows", "The number of rows.")
      .add_arg("cols", "The number of columns.")
      .add_arg("transpose_matrix", "Whether to transpose the matrix.")
      .add_arg("mma_M", "The M dimension of the matrix operation.")
      .add_arg("mma_N", "The N dimension of the matrix operation.")
      .add_arg("mma_K", "The K dimension of the matrix operation.")
      .add_arg("operand_role", "The matrix operand role.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cooperative_tensor_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cooperative_tensor_store")
      .add_arg("d", "The D operand.")
      .add_arg("index", "The index.")
      .add_arg("ptr", "The pointer.")
      .add_arg("stride", "The stride.")
      .add_arg("rows", "The number of rows.")
      .add_arg("cols", "The number of columns.")
      .add_arg("transpose_matrix", "Whether to transpose the matrix.")
      .add_arg("mma_M", "The M dimension of the matrix operation.")
      .add_arg("mma_N", "The N dimension of the matrix operation.")
      .add_arg("mma_K", "The K dimension of the matrix operation.")
      .add_arg("operand_role", "The matrix operand role.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cooperative_tensor_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cooperative_tensor_multiply_accumulate")
      .add_arg("d", "The D operand.")
      .add_arg("index_d", "The D fragment index.")
      .add_arg("a", "The A operand.")
      .add_arg("index_a", "The A fragment index.")
      .add_arg("b", "The B operand.")
      .add_arg("index_b", "The B fragment index.")
      .add_arg("c", "The C operand.")
      .add_arg("index_c", "The C fragment index.")
      .add_arg("M", "The M dimension.")
      .add_arg("N", "The N dimension.")
      .add_arg("K", "The K dimension.")
      .add_arg("transpose_a", "Whether to transpose A.")
      .add_arg("transpose_b", "Whether to transpose B.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("cooperative_tensor_multiply_accumulate"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.vectorhigh")
      .add_arg("vec", "The input vector.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("vectorhigh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.vectorlow")
      .add_arg("vec", "The input vector.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("vectorlow"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.vectorcombine")
      .add_arg("vec1", "The first input vector.")
      .add_arg("vec2", "The second input vector.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("vectorcombine"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.dp4a")
      .add_arg("vec1", "The first input vector.")
      .add_arg("vec2", "The second input vector.")
      .add_arg("acc", "The accumulator.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("dp4a"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.atomic_add")
      .add_arg("ptr", "The pointer.")
      .add_arg("value", "The value to use.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("atomic_add"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.nd_mem_alloc_with_scope")
      .add_arg("storage_scope", "The storage scope.")
      .add_arg("ndim", "The number of dimensions.")
      .add_arg("shape", "The shape.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("nd_mem_alloc_with_scope"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.texture2d_store")
      .add_arg("texture", "The texture.")
      .add_arg("x", "The input value.")
      .add_arg("y", "The second input value.")
      .add_arg("z", "The third input value.")
      .add_arg("channel_size", "The number of channels.")
      .add_arg("value", "The value to use.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("texture2d_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TVectorizable>("TVectorizable", true)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.texture2d_load")
      .add_arg("texture", "The texture.")
      .add_arg("x", "The input value.")
      .add_arg("y", "The second input value.")
      .add_arg("z", "The third input value.")
      .add_arg("channel_size", "The number of channels.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("texture2d_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TVectorizable>("TVectorizable", true)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.assume")
      .add_arg("cond", "The condition.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("assume"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kEmbedInfo));

  OpDef("tirx.undef")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("undef"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));

  OpDef("tirx.get_active_lane_mask")
      .add_arg("base", "The base value.")
      .add_arg("limit", "The limit value.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("get_active_lane_mask"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.masked_load")
      .add_arg("buffer", "The buffer.")
      .add_arg("index", "The index.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("masked_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.masked_store")
      .add_arg("buffer", "The buffer.")
      .add_arg("value", "The value to use.")
      .add_arg("index", "The index.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("masked_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kUpdateState));

  OpDef("tirx.ignore_loop_partition")
      .add_arg("predicate", "The predicate.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("ignore_loop_partition"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kNone));

  OpDef("tirx.buffer_offset")
      .add_arg("load", "The buffer load whose offset is returned.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("buffer_offset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.buffer_data")
      .add_arg("buffer", "The buffer.")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("buffer_data"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.print_buffer")
      .add_arg("data", "The input data.")
      .add_arg("dtype", "The data type.")
      .add_arg("is_string", "Whether to print as a string.")
      .add_arg("is_scalar", "Whether to print as a scalar.")
      .add_arg("ndim", "The number of dimensions.")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("print_buffer"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm

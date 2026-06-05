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
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {
namespace builtin {

#define TIR_DEFINE_BUILTIN_FUNC(OpName)             \
  const Op& OpName() {                              \
    static const Op& op = Op::Get("tirx." #OpName); \
    return op;                                      \
  }                                                 \
  TVM_TIRX_REGISTER_OP(#OpName)

TIR_DEFINE_BUILTIN_FUNC(reinterpret)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst))
    .set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(ret)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kControlJump))
    .set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(thread_return)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kControlJump))
    .set_num_inputs(0);

TIR_DEFINE_BUILTIN_FUNC(continue_loop)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kControlJump))
    .set_num_inputs(0);

TIR_DEFINE_BUILTIN_FUNC(break_loop)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kControlJump))
    .set_num_inputs(0);

TIR_DEFINE_BUILTIN_FUNC(likely)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kExprAnnotation))
    .set_attr<TVectorizable>("TVectorizable", true);

// tirx.filter: escape hatch for non-canonical thread-set filter predicates
// used as an IfThenElse condition. (var, cond) -- ``var`` names the
// active-set axis the compiler should collapse to a singleton if it cannot
// statically analyze ``cond``. Canonical predicates (see
// ``analysis/filter_canonical.h``) should appear bare in ``if`` conditions
// without this wrapper.
TIR_DEFINE_BUILTIN_FUNC(filter).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(selector).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(bitwise_and)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_or)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_xor)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_not)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(shift_left)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(shift_right)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(large_uint_imm)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(address_of)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(if_then_else)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(q_multiply_shift)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(q_multiply_shift_per_axis)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(isnullptr).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(isnan).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(popcount)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(fma)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(call_extern)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(call_pure_extern)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(call_llvm_intrin)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(call_llvm_pure_intrin)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(call_spirv_pure_glsl450)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(prefetch).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_access_ptr)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kSpecialCallArg));

TIR_DEFINE_BUILTIN_FUNC(ptr_byte_offset)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(tvm_static_handle)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kSpecialCallArg));

TIR_DEFINE_BUILTIN_FUNC(tvm_context_id)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kReadState));

TIR_DEFINE_BUILTIN_FUNC(tvm_tuple).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kEmbedInfo));

TIR_DEFINE_BUILTIN_FUNC(handle_add_byte_offset)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(tvm_struct_get)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kReadState))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kLast));

TIR_DEFINE_BUILTIN_FUNC(tvm_struct_set)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kUpdateState));

TIR_DEFINE_BUILTIN_FUNC(lookup_param)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kUpdateState));

TIR_DEFINE_BUILTIN_FUNC(tvm_throw_last_error)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_stack_alloca)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_stack_make_shape)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_stack_make_array)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

// When num_inputs are not set, the function is assumed to be variable length.
TIR_DEFINE_BUILTIN_FUNC(tvm_call_packed)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_packed"), /*plevel=*/20);

TIR_DEFINE_BUILTIN_FUNC(tvm_call_cpacked)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_cpacked"), /*plevel=*/20);

TIR_DEFINE_BUILTIN_FUNC(tvm_call_trace_packed)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_thread_invariant)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(tvm_call_packed_lowered)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_packed_lowered"),
                                  /*plevel=*/20);

TIR_DEFINE_BUILTIN_FUNC(tvm_call_cpacked_lowered)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("call_cpacked_lowered"),
                                  /*plevel=*/20);

TIR_DEFINE_BUILTIN_FUNC(tvm_call_trace_packed_lowered)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

// TODO(tvm-team) revisit storage sync once we have a good memory hierachy structure.
TIR_DEFINE_BUILTIN_FUNC(tvm_storage_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_kernel_replace_point)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_shuffle)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_shuffle_up)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_shuffle_down)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_shuffle_xor)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_activemask)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_global_barrier_kinit)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_thread_allreduce)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(make_filled_simdgroup_matrix)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(simdgroup_load)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(simdgroup_store)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(simdgroup_multiply_accumulate)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cooperative_tensor_fill)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cooperative_tensor_load)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cooperative_tensor_store)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cooperative_tensor_multiply_accumulate)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(vectorhigh)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(vectorlow)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(vectorcombine)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(dp4a)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(atomic_add)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(nd_mem_alloc_with_scope)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(texture2d_store)
    .set_attr<TVectorizable>("TVectorizable", true)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(texture2d_load)
    .set_attr<TVectorizable>("TVectorizable", true)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(dma_copy).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(dma_wait).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(dma_start_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(dma_end_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(assume)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kEmbedInfo))
    .set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(undef)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kReadState))
    .set_num_inputs(0);

TIR_DEFINE_BUILTIN_FUNC(start_profile_intrinsic)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(end_profile_intrinsic)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(anylist_getitem)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kReadState));

TIR_DEFINE_BUILTIN_FUNC(anylist_resetitem)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "TVMBackendAnyListResetItem");

TIR_DEFINE_BUILTIN_FUNC(anylist_setitem_call_packed)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(anylist_setitem_call_cpacked)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(vscale).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(get_active_lane_mask)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(ignore_loop_partition)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kNone));
TIR_DEFINE_BUILTIN_FUNC(buffer_offset)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(print_buffer)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(timer_init_cuda)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(timer_start_cuda)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(timer_end_cuda)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(timer_finalize_cuda)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_atomic_add)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_thread_fence)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_warpgroup_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_warp_reduce)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_cta_reduce)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_copy_bytes)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_warp_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_cta_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_grid_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_thread_rank)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

// Cluster-wide sync (CUDA thread block clusters)
TIR_DEFINE_BUILTIN_FUNC(cuda_cluster_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_half2float)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_bfloat162float)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_float22half2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_trap_when_assert_failed)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_runtime_instr_desc)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_half8tofloat8)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_float8tohalf8)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_syncthreads_and)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_syncthreads_or)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_nano_sleep)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_atomic_cas)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_printf)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_ldg)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_num_inputs(2);

TIR_DEFINE_BUILTIN_FUNC(cuda_get_tmem_addr)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_exp2).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(ptx_rcp).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(ptx_any_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(ptx_reduce3_max_f32)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(ptx_reduce3_min_f32)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

// PTX scalar / packed floating-point arithmetic, DPS form (writes to *d_addr).
//   add/sub/mul: 2 sources, 1 destination.
//   fma:         3 sources, 1 destination.
//   Modifiers (rounding / ftz / sat) are codegen attrs.
// kOpaque because all four kinds write through the destination pointer.
TIR_DEFINE_BUILTIN_FUNC(ptx_add_f32)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
TIR_DEFINE_BUILTIN_FUNC(ptx_add_f32x2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
TIR_DEFINE_BUILTIN_FUNC(ptx_add_f64)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_sub_f32)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
TIR_DEFINE_BUILTIN_FUNC(ptx_sub_f32x2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
TIR_DEFINE_BUILTIN_FUNC(ptx_sub_f64)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_mul_f32)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
TIR_DEFINE_BUILTIN_FUNC(ptx_mul_f32x2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
TIR_DEFINE_BUILTIN_FUNC(ptx_mul_f64)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_fma_f32)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
TIR_DEFINE_BUILTIN_FUNC(ptx_fma_f32x2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
TIR_DEFINE_BUILTIN_FUNC(ptx_fma_f64)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

// max stays value-returning + kPure (no .sat, not in the add/sub/mul/fma family).
TIR_DEFINE_BUILTIN_FUNC(ptx_max_f32)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}  // namespace builtin
}  // namespace tirx
}  // namespace tvm

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
 * \file tir/op/builtin.cc
 *
 *  builtin intrinsic operators.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tir {
namespace builtin {

#define TIR_DEFINE_BUILTIN_FUNC(OpName)            \
  const Op& OpName() {                             \
    static const Op& op = Op::Get("tir." #OpName); \
    return op;                                     \
  }                                                \
  TVM_REGISTER_OP("tir." #OpName)

TIR_DEFINE_BUILTIN_FUNC(reinterpret)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(ret)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kControlJump))
    .set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(likely)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kExprAnnotation))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_and)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_or)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_xor)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_not)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(shift_left)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(shift_right)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(large_uint_imm)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(address_of)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(if_then_else)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(q_multiply_shift)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(q_multiply_shift_per_axis)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(isnullptr).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(isnan).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(popcount)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(fma)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(call_extern)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(call_pure_extern)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(call_llvm_intrin)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(call_llvm_pure_intrin)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(call_spirv_pure_glsl450)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(prefetch).set_attr<TCallEffectKind>("TCallEffectKind",
                                                            Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_access_ptr)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kSpecialCallArg));

TIR_DEFINE_BUILTIN_FUNC(tvm_static_handle)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kSpecialCallArg));

TIR_DEFINE_BUILTIN_FUNC(tvm_context_id)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kReadState));

TIR_DEFINE_BUILTIN_FUNC(tvm_tuple).set_attr<TCallEffectKind>("TCallEffectKind",
                                                             Integer(CallEffectKind::kEmbedInfo));

TIR_DEFINE_BUILTIN_FUNC(tvm_struct_get)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kReadState));

TIR_DEFINE_BUILTIN_FUNC(tvm_struct_set)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kUpdateState));

TIR_DEFINE_BUILTIN_FUNC(lookup_param)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kUpdateState));

TIR_DEFINE_BUILTIN_FUNC(tvm_throw_last_error)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_stack_alloca)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_stack_make_shape)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_stack_make_array)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

// When num_inputs are not set, the function is assumed to be variable length.
TIR_DEFINE_BUILTIN_FUNC(tvm_call_packed)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_call_cpacked)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_call_trace_packed)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_check_return)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(tvm_thread_context)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_call_packed_lowered)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_call_cpacked_lowered)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_call_trace_packed_lowered)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

// TODO(tvm-team) revisit storage sync once we have a good memory hierachy structure.
TIR_DEFINE_BUILTIN_FUNC(tvm_storage_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_shuffle)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_shuffle_up)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_shuffle_down)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_activemask)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_global_barrier_kinit)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_thread_allreduce)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_load_matrix_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kReadState));

TIR_DEFINE_BUILTIN_FUNC(tvm_mma_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_bmma_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_fill_fragment)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_store_matrix_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_mma).set_attr<TCallEffectKind>("TCallEffectKind",
                                                           Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_mma_sp)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_ldmatrix)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_cp_async)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_commit_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_wait_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(mma_store).set_attr<TCallEffectKind>("TCallEffectKind",
                                                             Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(mma_fill).set_attr<TCallEffectKind>("TCallEffectKind",
                                                            Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(vectorhigh)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(vectorlow).set_attr<TCallEffectKind>("TCallEffectKind",
                                                             Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(vectorcombine)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(atomic_add)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(nd_mem_alloc_with_scope)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(texture2d_store)
    .set_attr<TVectorizable>("TVectorizable", true)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(texture2d_load)
    .set_attr<TVectorizable>("TVectorizable", true)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(dma_copy).set_attr<TCallEffectKind>("TCallEffectKind",
                                                            Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(dma_wait).set_attr<TCallEffectKind>("TCallEffectKind",
                                                            Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(assume)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kEmbedInfo))
    .set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(undef)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kReadState))
    .set_num_inputs(0);

TIR_DEFINE_BUILTIN_FUNC(start_profile_intrinsic)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(end_profile_intrinsic)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

}  // namespace builtin
}  // namespace tir
}  // namespace tvm

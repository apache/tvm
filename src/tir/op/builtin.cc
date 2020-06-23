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

TIR_DEFINE_BUILTIN_FUNC(reinterpret).set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(likely).set_num_inputs(1).set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_and)
    .set_num_inputs(2)
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_or)
    .set_num_inputs(2)
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_xor)
    .set_num_inputs(2)
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(bitwise_not)
    .set_num_inputs(1)
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(shift_left)
    .set_num_inputs(2)
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(shift_right)
    .set_num_inputs(2)
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(large_uint_imm).set_num_inputs(2);

TIR_DEFINE_BUILTIN_FUNC(address_of).set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(if_then_else).set_num_inputs(3);

TIR_DEFINE_BUILTIN_FUNC(isnullptr).set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(isnan).set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(popcount).set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(fma).set_num_inputs(3).set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_BUILTIN_FUNC(call_extern);

TIR_DEFINE_BUILTIN_FUNC(call_llvm_intrin);

TIR_DEFINE_BUILTIN_FUNC(call_spirv_glsl450);

TIR_DEFINE_BUILTIN_FUNC(prefetch);

TIR_DEFINE_BUILTIN_FUNC(tvm_access_ptr).set_num_inputs(5);

TIR_DEFINE_BUILTIN_FUNC(tvm_static_handle).set_num_inputs(0);

TIR_DEFINE_BUILTIN_FUNC(tvm_context_id).set_num_inputs(0);

TIR_DEFINE_BUILTIN_FUNC(tvm_tuple);

TIR_DEFINE_BUILTIN_FUNC(tvm_struct_get).set_num_inputs(3);

TIR_DEFINE_BUILTIN_FUNC(tvm_struct_set).set_num_inputs(4);

TIR_DEFINE_BUILTIN_FUNC(tvm_throw_last_error).set_num_inputs(0);

TIR_DEFINE_BUILTIN_FUNC(tvm_stack_alloca).set_num_inputs(2);

TIR_DEFINE_BUILTIN_FUNC(tvm_stack_make_shape);

TIR_DEFINE_BUILTIN_FUNC(tvm_stack_make_array).set_num_inputs(6);

// When num_inputs are not set, the function is assumed to be variable length.
TIR_DEFINE_BUILTIN_FUNC(tvm_call_packed);

TIR_DEFINE_BUILTIN_FUNC(tvm_call_trace_packed);

TIR_DEFINE_BUILTIN_FUNC(tvm_thread_context).set_num_inputs(1);

TIR_DEFINE_BUILTIN_FUNC(tvm_call_packed_lowered);

TIR_DEFINE_BUILTIN_FUNC(tvm_call_trace_packed_lowered);

// TODO(tvm-team) revisit storage sync once we have a good memory hierachy structure.
TIR_DEFINE_BUILTIN_FUNC(tvm_storage_sync);

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_shuffle);

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_shuffle_up);

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_shuffle_down);

TIR_DEFINE_BUILTIN_FUNC(tvm_warp_activemask);

TIR_DEFINE_BUILTIN_FUNC(tvm_global_barrier_kinit);

TIR_DEFINE_BUILTIN_FUNC(tvm_thread_allreduce);

TIR_DEFINE_BUILTIN_FUNC(tvm_load_matrix_sync);

TIR_DEFINE_BUILTIN_FUNC(tvm_mma_sync);

TIR_DEFINE_BUILTIN_FUNC(tvm_bmma_sync);

TIR_DEFINE_BUILTIN_FUNC(tvm_fill_fragment);

TIR_DEFINE_BUILTIN_FUNC(tvm_store_matrix_sync);

TIR_DEFINE_BUILTIN_FUNC(vectorhigh);

TIR_DEFINE_BUILTIN_FUNC(vectorlow);

TIR_DEFINE_BUILTIN_FUNC(vectorcombine);

}  // namespace builtin
}  // namespace tir
}  // namespace tvm

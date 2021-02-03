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
 * \file tvm/tir/builtin.h
 * \brief TIR builtin intrinsics.
 *
 * TIR builtin intrinsics are stored as tvm:Op.
 * They are processed in the same way as we process Ops.
 *
 * It is not necessary to create a function for every Op,
 * as we can obtain them through Op::Get.
 *
 * This file contains the most commonly used intrinsics or
 * those that have special semantics and need compiler support.
 */
#ifndef TVM_TIR_BUILTIN_H_
#define TVM_TIR_BUILTIN_H_

#include <tvm/ir/op.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace tir {

/*! \brief Collection of builtin intrinsics as ops */
namespace builtin {
/*!
 * \brief Return value.
 */
TVM_DLL const Op& ret();
/*!
 * \brief Reinterpret the value using the target type.
 */
TVM_DLL const Op& reinterpret();

/*!
 * \brief Marks a condition is likely going to happen.
 */
TVM_DLL const Op& likely();

/*!
 * \brief Bitwise and operator.
 */
TVM_DLL const Op& bitwise_and();

/*!
 * \brief Bitwise or operator.
 */
TVM_DLL const Op& bitwise_or();

/*!
 * \brief Bitwise xor operator.
 */
TVM_DLL const Op& bitwise_xor();

/*!
 * \brief Bitwise not operator.
 */
TVM_DLL const Op& bitwise_not();

/*!
 * \brief Left shift
 */
TVM_DLL const Op& shift_left();

/*!
 * \brief Right shift
 */
TVM_DLL const Op& shift_right();

/*!
 * \brief See pesudo code
 *
 *  Construct a big uint that may not be representable by int64
 *
 *  Expr large_uint_imm(uint32_t v0, uin32_t v1) {
 *    return (v1 << 32) | v0;
 *  }
 */
TVM_DLL const Op& large_uint_imm();

/*!
 * \brief Execute a multiplication between two Q-numbers x and y
 * followed by a right shift s
 * The default rounding rule is to the nearest value, rounding half up
 * (i.e., round(x.1) = x and round (x.5) = x+1)
 */
TVM_DLL const Op& q_multiply_shift();

/*!
 * \brief See pesudo code
 *
 *  Handle address_of(Load *op) {
 *     return &op->buffer_var[index];
 *  }
 */
TVM_DLL const Op& address_of();

/*!
 * \brief Same as select, used for unsafe memory access.
 *
 *  Type tvm_if_then_else(cond, a, b) {
 *    return cond ? a : b;
 *  }
 */
TVM_DLL const Op& if_then_else();

/*!
 * \brief See pesudo code
 *
 *  bool isnullptr(void* handle) {
 *     return handle == nullptr
 *  }
 */
TVM_DLL const Op& isnullptr();

/*!
 * \brief Check if value is nan
 */
TVM_DLL const Op& isnan();

/*!
 * \brief Popcount
 */
TVM_DLL const Op& popcount();

/*!
 * \brief Fused multiply add
 *
 *  Type fma(a, b, c) {
 *    return a * b + c;
 *  }
 */
TVM_DLL const Op& fma();

/*!
 * \brief Call an extern C function with given name
 *        and signature from the types of args in the runtime environment.
 *
 *  Type call_extern(name, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This intrinsic does not provide any type checking,
 *       and is main used for backward compatibility reasons.
 *       Always consider use pre-registered and typed tvm::Op first.
 */
TVM_DLL const Op& call_extern();

/*!
 * \brief Call an pure extern C function with given name
 *        and signature from the types of args in the runtime environment.
 *
 *  Type call_pure_extern(name, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This intrinsic does not provide any type checking,
 *       and is main used for backward compatibility reasons.
 *       Always consider use pre-registered and typed tvm::Op first.
 */
TVM_DLL const Op& call_pure_extern();

/*!
 * \brief Call an LLVM intrinsic with a given intrinsic id
 *        and signature from the types of args in the runtime environment.
 *
 *  Type call_llvm_pure_intrin(intrin_id, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This op does not provide any type checking.
 */
TVM_DLL const Op& call_llvm_intrin();

/*!
 * \brief Call an LLVM pure intrinsic with a given intrinsic id
 *        and signature from the types of args in the runtime environment.
 *
 *  Type call_llvm_pure_intrin(intrin_id, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This op does not provide any type checking.
 */
TVM_DLL const Op& call_llvm_pure_intrin();

/*!
 * \brief Call an SPIRV pure GLSL450 intrinsic.
 *
 *  Type call_spirv_pure_glsl450(intrin_id, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This op does not provide any type checking.
 */
TVM_DLL const Op& call_spirv_pure_glsl450();

// TODO(tvm-team) revisit the builtins below
// some of them can simply become ops with special codegen attr.
/*!
 * \brief Prefetch a cacheline
 */
TVM_DLL const Op& prefetch();

/*!
 * \brief Get head access address with memory access pattern info.
 *
 *  This operator also marks range of the memory access
 *  The offset and extent are in unit of the DType(including vectorization factor).
 *  rw_mask is a bit_mask setting whether the access is a read(1) or write(2).
 *  The access is assume to happen in the current expression.
 *
 *  PtrType tvm_access_ptr(Expr dtype, DType* data,
 *                         int offset, int extent,
 *                         int rw_mask) {
 *    // DType == dtype.type();
 *    return &data[offset];
 *  }
 */
TVM_DLL const Op& tvm_access_ptr();

/*!
 * \brief Create a function local static handle that iniitalizes to nullptr.
 *  can be used to cache function local static resources.
 */
TVM_DLL const Op& tvm_static_handle();

/*!
 * \brief Return a unique context id, used for hint of workspace separation.
 *  Different context id ganrantees not having overlapping workspace.
 */
TVM_DLL const Op& tvm_context_id();

/*!
 * \brief tvm_tuple is not an actual function and cannot codegen.
 *  It is used to represent tuple structure in value field of AttrStmt,
 *  for the sake of giving hint to optimization.
 *
 *  Handle tvm_tuple(value0, value1, ..., value_n);
 */
TVM_DLL const Op& tvm_tuple();

/*!
 * \brief See pesudo code
 *
 *  Type tvm_struct_get(StructType* arr, int index, int field_id) {
 *     return arr[index]->field;
 *  }
 * \sa TVMStructFieldKind
 */
TVM_DLL const Op& tvm_struct_get();

/*!
 * \brief See pesudo code
 *
 *  Handle tvm_struct_set(StructType* arr, int index, int field_id, value) {
 *     arr[index]->field = value;
 *  }
 * \sa TVMStructFieldKind
 */
TVM_DLL const Op& tvm_struct_set();

/*!
 * \brief See pesudo code
 *
 *  void tvm_throw_last_error() {
 *    throw TVMGetLastError();
 *  }
 */
TVM_DLL const Op& tvm_throw_last_error();

/*!
 * \brief See pesudo code
 *
 *  dtype in {shape, array, arg_value, arg_tcode}
 *
 *  Handle tvm_stack_alloca(string dtype, int num) {
 *     return new on stack dtype[num];
 *  }
 */
TVM_DLL const Op& tvm_stack_alloca();

/*!
 * \brief Allocate a shape tuple on stack, return the handle.
 *
 *  Handle tvm_stack_make_shape(list args) {
 *     ret = alloca stack int64_t[len(args)];
 *     for i in range(len(args)):
 *        ret[i] = args[i]
 *     return &ret[0];
 *  }
 */
TVM_DLL const Op& tvm_stack_make_shape();

/*!
 * \brief Allocate a NDArray(DLTensor) on stack, return the handle.
 *
 *  Type tvm_stack_make_array(Expr data,
 *                            Expr shape,
 *                            Expr strides,
 *                            Expr ndim,
 *                            Expr dtype,
 *                            Expr elem_offset) {
 *     ret = alloca stack DLTensor();
 *     ret->data = data;
 *     ret->shape = shape;
 *     ret->strides = strides != 0 ? strides : nullptr;
 *     ret->ndim = ndim;
 *     ret->dtype = dtype.type();
 *     ret->byte_offset = elem_offset * sizeof(dtype);
 *     return ret;
 *  }
 */
TVM_DLL const Op& tvm_stack_make_array();

/*!
 * \brief See pesudo code
 *
 *  int tvm_call_packed(name, TVMValue* args) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     (*f)(args, type_code_of(args), len(args));
 *     return 0;
 *  }
 */
TVM_DLL const Op& tvm_call_packed();

/*!
 * \brief See pesudo code
 *
 *  int tvm_call_trace_packed(name, TVMValue* args) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     (*f)(args, type_code_of(args), len(args));
 *     return 0;
 *  }
 */
TVM_DLL const Op& tvm_call_trace_packed();

/*!
 * \brief See pesudo code
 *  Mark the content as thread local context, can get optimized
 *  by only call the call once at thread start.
 *
 *  Do not allow nesting(getting a thread context from another).
 *
 *  Handle tvm_thread_context(Expr call) {
 *     return call;
 *  }
 */
TVM_DLL const Op& tvm_thread_context();

/*!
 * \brief Lowered version of call packed, the space of value and
 *  type codes are explicitly allocated.
 *
 *  int tvm_call_packed_lowered(name,
 *                              TVMValue* value_stack,
 *                              int* tcode_stack,
 *                              int begin,
 *                              int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(TVMArgs(value_stack[begin:end],
 *                           tcode_stack[begin:end]),
 *                   TVMRetValue(value_stack + end, tcode_stack + end));
 *  }
 */
TVM_DLL const Op& tvm_call_packed_lowered();

/*!
 * \brief Lowered version of trace intrinsic, the space of value and
 *  type codes are explicitly allocated. The return value is the
 *  (end - 1) value on the stack.
 *
 *  int tvm_call_trace_packed_lowered(name,
 *                                    TVMValue* value_stack,
 *                                    int* tcode_stack,
 *                                    int begin,
 *                                    int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(TVMArgs(value_stack[begin:end],
 *                           tcode_stack[begin:end]),
 *                   TVMRetValue(value_stack + end, tcode_stack + end));
 *  }
 */
TVM_DLL const Op& tvm_call_trace_packed_lowered();

/*!
 * \brief See pseudo code
 *
 *  int tvm_storage_sync(std::string storage_scope) {
 *     __sync(storage_scope);
 *     return 0;
 *  }
 */
TVM_DLL const Op& tvm_storage_sync();

/*!
 * \brief See pseudo code
 *
 *  Type tvm_warp_shuffle(mask, Type value, warp_id, width, warp_size) {
 *    return (value passed in by warp indicated by this_warp_id);
 *  }
 *
 *  Type tvm_warp_shuffle_up(mask, Type value, offset, width, warp_size) {
 *    return (value passed in by warp indicated by this_warp_id - offset);
 *  }
 *
 *  Type tvm_warp_shuffle_down(mask, Type value, offset, width, warp_size) {
 *    return (value passed in by warp indicated by this_warp_id + offset);
 *  }
 *
 *  unsigned tvm_warp_activemask() {
 *    return (32-bit mask of currently active threads in the calling warp);
 *  }
 *
 *  Parameter warp_id indicates the source thread ID in a warp.
 *
 *  Parameter offset indicates the relative distance to this_warp_id.
 *
 *  Parameter width indicates the number of threads involved in one
 *  shuffle. See CUDA document for __shfl_sync, __shfl_up_sync,
 *  __shfl_down_sync and __activemask.
 *
 *  Parameter warp_size is the size of a warp, which helps a backend
 *  to determine wheter the width paramter is legal.
 *
 */
TVM_DLL const Op& tvm_warp_shuffle();
TVM_DLL const Op& tvm_warp_shuffle_up();
TVM_DLL const Op& tvm_warp_shuffle_down();
TVM_DLL const Op& tvm_warp_activemask();

/*!
 * \brief Initialize the global barrier.
 *  Call this at beginning of kernel that need global barrier.
 */
TVM_DLL const Op& tvm_global_barrier_kinit();

/*!
 * \brief See pesudo code
 *
 *  void tvm_thread_allreduce(UIntImm size, Expr source0, ..., Expr cond,
 *                            Var reduce_temp0, .., Var thread_idx1, ...) {
 *     // constraint by the other thread_idx remain the same.
 *     // reduce_temp is used to save intermediate result.
 *     reduce_temp0, ... = reduce(combiner, source0, ..., cond
 *       over [thread_idx1, thread_idx2] passed by any caller)
 *  }
 */
TVM_DLL const Op& tvm_thread_allreduce();

// TODO(tvm-team) TensorCore specific intrinsics should be directly registered under
//                cuda. namespace and used through op.
/*!
 * \brief tvm intrinsic for tensor core load operators.
 *
 *  void tvm_load_matrix_sync(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
 *                            Expr index, Expr buffer_ptr, Expr stride,
 *                            StringImm layout) {
 *    // m, n, k are the shape of wmma fragment.
 *    // Determine fragment layout(column-major or row major) by layout.
 *    // fragments must be in 'wmma.matrix_a' or 'wmma.matrix_b' scope.
 *    nvcuda::wmma::load_matrix_sync(fragment[index], buffer_ptr, stride);
 *  }
 */
TVM_DLL const Op& tvm_load_matrix_sync();

/*!
 * \brief tvm intrinsic for tensor core mma_sync operators.
 *
 *  void tvm_mma_sync(Var fragment_d, Expr index_d,
 *                    Var fragment_a, Expr index_a,
 *                    Var fragment_b, Expr index_b,
 *                    Var fragment_c, Expr index_c) {
 *    nvcuda::wmma::mma_sync(fragment_d[index_d], fragment_a[index_a],
 *                           fragment_b[index_b], fragment_c[index_c]);
 *  }
 */
TVM_DLL const Op& tvm_mma_sync();

/*!
 * \brief tvm intrinsic for tensor core bmma_sync operators.
 *
 *  void tvm_bmma_sync(Var fragment_d, Expr index_d,
 *                     Var fragment_a, Expr index_a,
 *                     Var fragment_b, Expr index_b,
 *                     Var fragment_c, Expr index_c) {
 *    nvcuda::wmma::bmma_sync(fragment_d[index_d], fragment_a[index_a],
 *                           fragment_b[index_b], fragment_c[index_c]);
 *  }
 */
TVM_DLL const Op& tvm_bmma_sync();

/*!
 * \brief tvm intrinsic for tensor core fill_fragment operators.
 *
 *  void tvm_fill_fragment(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
 *                         Expr index, Expr value) {
 *    // m, n, k are the shape of wmma fragment
 *    // fragments must be in 'wmma.accumulator' scope.
 *    nvcuda::wmma::fill_fragment(fragment[index], value);
 *  }
 */
TVM_DLL const Op& tvm_fill_fragment();

/*!
 * \brief tvm intrinsic for tensor core store operators.
 *
 *  void tvm_store_matrix_sync(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
 *                             Expr index, Expr buffer_ptr, Expr stride,
 *                             StringImm layout) {
 *    // m, n, k are the shape of wmma fragment
 *    // fragments must be in 'wmma.accumulator' scope.
 *    nvcuda::wmma::store_matrix_sync(fragment[index], buffer_ptr, stride, layout);
 *  }
 */
TVM_DLL const Op& tvm_store_matrix_sync();

// TODO(tvm-team) replace the usage of the vector operations by Shuffle.
/*!
 * \brief Get the high level half of the vector
 */
TVM_DLL const Op& vectorhigh();

/*!
 * \brief Get the low-level half of the vector
 */
TVM_DLL const Op& vectorlow();

/*!
 * \brief Concat two vectors.
 */
TVM_DLL const Op& vectorcombine();

/*!
 * \brief atomic add instruction, corresponding e.g. to atomicAdd in CUDA
 */
TVM_DLL const Op& atomic_add();

/*! \brief The kind of structure field info used in intrinsic */
enum TVMStructFieldKind : int {
  // array head address
  kArrAddr,
  kArrData,
  kArrShape,
  kArrStrides,
  kArrNDim,
  kArrTypeCode,
  kArrTypeBits,
  kArrTypeLanes,
  kArrByteOffset,
  kArrDeviceId,
  kArrDeviceType,
  kArrKindBound_,
  // TVMValue field
  kTVMValueContent,
  kTVMValueKindBound_
};
}  // namespace builtin
}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_BUILTIN_H_

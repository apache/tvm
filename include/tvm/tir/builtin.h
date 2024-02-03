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
 * \brief Returns the address of an element in the buffer (see pseudocode below).
 *
 * The number of indices should match the dimensionality of the buffer
 * being accessed.  If this operation occurs after buffer flattening,
 * the number of indices must be supported by the target (i.e. N>1
 * only on targets that support non-flat memory buffers).
 *
 *  Handle address_of(BufferLoad *op) {
 *     return &op->buffer_var[op->indices[0], op->indices[1], ..., op->indices[N-1]];
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
 * \brief See pseudo code
 * Type lookup_param(String param_name) {
 *     return __tvm_param__param_name;
 * }
 */
TVM_DLL const Op& lookup_param();

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
 *  return_type tvm_call_packed(name, TVMValue* args) {
 *     TVMValue ret_value;
 *     int ret_code;
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     (*f)(args, type_code_of(args), len(args), &ret_value, &ret_code);
 *     // return type can be int, float, handle.
 *     return cast(return_type, ret_value.v_return_type);
 *  }
 */
TVM_DLL const Op& tvm_call_packed();

/*!
 * \brief See pesudo code
 *
 * return_type tvm_call_packed(fname, TVMValue* args) {
 * 	   int ret_code;
 *     TVMValue ret_value;
 *     (*fname)(args, type_code_of(args), len(args), &ret_value, &ret_code);
 *     return cast(return_type, ret_value.v_return_type);
 *  }
 */
TVM_DLL const Op& tvm_call_cpacked();

/*!
 * \brief See pesudo code
 *
 *  return_type tvm_call_trace_packed(name, TVMValue* args) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     (*f)(args, type_code_of(args), len(args));
 *     // return type can be int, float, handle.
 *     return cast(return_type, ret_value.v_return_type);
 *  }
 */
TVM_DLL const Op& tvm_call_trace_packed();

/*!
 * \brief Checks the return value of another call is correct or returns a given value.
 *
 * \note  This is meant to serve a specific case for AOT code generator whilst this
 *        cannot be fully represented in TIR.
 *
 *  Type tvm_check_return(expected, return_unexpected, nested_call) {
 *     if (nested_call() != expected) {
 *         return return_unexpected;
 *     }
 *  }
 */
TVM_DLL const Op& tvm_check_return();

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
 * \brief Mark a condition to be thread invariant.
 *  This means the condition must be the same for all threads.
 */
TVM_DLL const Op& tvm_thread_invariant();

/*!
 * \brief Lowered version of call packed, the space of value and
 *  type codes are explicitly allocated.
 *
 *  return_type tvm_call_packed_lowered(name,
 *                                      TVMValue* value_stack,
 *                                      int* tcode_stack,
 *                                      int begin,
 *                                      int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(TVMArgs(value_stack[begin:end],
 *                           tcode_stack[begin:end]),
 *                   TVMRetValue(value_stack + end, tcode_stack + end));
 *     // return type can be int, float, handle.
 *     return cast(return_type, load_return_from(tcode_stack + end))
 *  }
 */
TVM_DLL const Op& tvm_call_packed_lowered();

/*!
 * \brief Lowered version of call c-packed, the space of value and
 *  type codes are explicitly allocated.
 *
 *  int tvm_call_packed_lowered(fname,
 *                              TVMValue* value_stack,
 *                              int* tcode_stack,
 *                              int begin,
 *                              int end) {
 *     fname(TVMArgs(value_stack[begin:end], tcode_stack[begin:end]),
 *                   TVMRetValue(value_stack + end, tcode_stack + end));
 *  }
 */
TVM_DLL const Op& tvm_call_cpacked_lowered();

/*!
 * \brief Lowered version of trace intrinsic, the space of value and
 *  type codes are explicitly allocated. The return value is the
 *  (end - 1) value on the stack.
 *
 *  return_type tvm_call_trace_packed_lowered(name,
 *                                            TVMValue* value_stack,
 *                                            int* tcode_stack,
 *                                            int begin,
 *                                            int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(TVMArgs(value_stack[begin:end],
 *                           tcode_stack[begin:end]),
 *                   TVMRetValue(value_stack + end, tcode_stack + end));
 *     // return type can be int, float, handle.
 *     return cast(return_type, load_return_from(tcode_stack + end))
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

/*!
 * \brief tvm intrinsic for ptx tensor core mma instructions.
 *
 *  void ptx_mma(StringImm shape, StringImm A_layout, StringImm B_layout,
 *               StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *               Var multiplicand_a, Expr a_index,
 *               Var multiplicand_b, Expr b_index,
 *               Var accumulator, Expr c_index, bool saturate);
 */
TVM_DLL const Op& ptx_mma();

/*!
 * \brief tvm intrinsic for ptx predicate load with 32-bit data type.
 *
 */
TVM_DLL const Op& ptx_ldg32();

/*!
 * \brief tvm intrinsic for ptx predicate load with 32-bit data type.
 *
 */
TVM_DLL const Op& ptx_ldg32();

/*!
 * \brief tvm intrinsic for sparse tensor core ptx instructions.
 *
 * void ptx_mma_sp(StringImm shape, StringImm A_layout, StringImm B_layout,
 *                 StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *                 Var multiplicand_a, Expr a_index,
 *                 Var multiplicand_b, Expr b_index,
 *                 Var accumulator, Expr c_index,
 *                 Var metadata, Expr meta_index,
 *                 Var sparse_selector, bool saturate);
 */
TVM_DLL const Op& ptx_mma_sp();

/*!
 * \brief tvm intrinsic for ptx load matrix from shared memory.
 *
 * void ptx_ldmatrix(Bool trans, IntImm num, StringImm type,
 *                   Var local_ptr, Expr local_offset,
 *                   Var smem_ptr, Expr smem_offset);
 */
TVM_DLL const Op& ptx_ldmatrix();

/*!
 * \brief tvm intrinsics for ptx async copy from global to shared memory using cp.async
 *
 * void ptx_cp_async(Var shared_ptr,
 *                   Expr shared_offset,
 *                   Var global_ptr,
 *                   Expr global_offset,
 *                   size_t bytes);
 */
TVM_DLL const Op& ptx_cp_async();

/*!
 * \brief tvm intrinsics for ptx async copy from global to shared memory using cp.async.bulk
 *
 * void ptx_cp_async(Var shared_ptr,
 *                   Expr shared_offset,
 *                   Var global_ptr,
 *                   Expr global_offset,
 *                   size_t bytes,
 *                   int barrier_id);
 */
TVM_DLL const Op& ptx_cp_async_bulk();

/*!
 * \brief tvm intrinsics for ptx async copy commit and wait.
 *
 * void ptx_commit_group();
 * void ptx_wait_group(int num);
 *
 */
TVM_DLL const Op& ptx_commit_group();
TVM_DLL const Op& ptx_wait_group();

/*!
 * \brief tvm intrinsics for ptx async copy barrier using cp.async.mbarrier.arrive
 *
 * ptx_cp_async_barrier(int barrier_id)
 *
 */
TVM_DLL const Op& ptx_cp_async_barrier();

/*!
 * \brief tvm intrinsics for ptx barrier initialization of thread count using mbarrier.init
 *
 * ptx_init_barrier_thread_count(int barrier_id, int thread_count)
 *
 */
TVM_DLL const Op& ptx_init_barrier_thread_count();

/*!
 * \brief tvm intrinsics for ptx barrier arrival using mbarrier.arrive
 *
 * ptx_arrive_barrier(int barrier_id)
 *
 */
TVM_DLL const Op& ptx_arrive_barrier();

/*!
 * \brief tvm intrinsic for ptx barrier arrival with expect tx using mbarrier.arrive.expect_tx
 *
 * ptx_arrive_barrier_expect_tx(int barrier_id, int byte_count)
 *
 */
TVM_DLL const Op& ptx_arrive_barrier_expect_tx();

/*!
 * \brief tvm intrinsics for ptx barrier wait using mbarrier.try_wait
 *
 * ptx_wait_barrier(int barrier_id)
 *
 */
TVM_DLL const Op& ptx_wait_barrier();

/*!
 * \brief tvm intrinsics to create N barriers
 *
 * ptx_wait_barrier(int barrier_count)
 *
 */
TVM_DLL const Op& create_barriers();

/*!
 * \brief tvm intrinsic for storing the result of PTX MMA into a destination pointer.
 *        For example, if each thread in a warp of size 32 has 4 elements from the result of
 *        m16xn8xk16 MMA in its registers, this intrinsic can be used to store the result in a
 *        16x8 region in shared or global memory.
 *
 *        There is no real PTX instruction that does that, but we want to hide details of
 *        complex index manipulation behind this intrinsic to simplify TIR lowering passes (e.g.
 *        LowerWarpMemory).
 *
 * void mma_store(IntImm m, IntImm n, Var dst_ptr, Var src_ptr, Expr src_offset, Var dst_stride);
 */
TVM_DLL const Op& mma_store();

/*!
 * \brief tvm intrinsic for zero-initalizing an MMA accumulation registor.
 *        For example, if each thread in a warp of size 32 has 8 elements from the A matrix in
 *        m16xn8xk16 MMA in its registers, this intrinsic can be used to zero-initialize its
 *        4 accumulation registers.
 *
 *        There is no real PTX instruction that does that, but we introduce this intrinsic for the
 *        same reason as mma_store above.
 *
 * void mma_fill(IntImm local_size, Var local_ptr, Expr offset);
 */
TVM_DLL const Op& mma_fill();

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
/*!
 * \brief Create an Nd memory allocation with storage scope
 */
TVM_DLL const Op& nd_mem_alloc_with_scope();

/*!
 * \brief Store to texture 2d memory
 */
TVM_DLL const Op& texture2d_store();

/*!
 * \brief Load from texture 2d memory
 */
TVM_DLL const Op& texture2d_load();

/*!
 * \brief Initiate a non-blocking DMA copy from source to destination
 *
 * The copy is launched immediately.
 *
 * If a `dma_start_group()` call is active, the copy will be added
 * to the current group for tracking of in-flight group counts.
 *
 * If no `dma_start_group()` call is active, the copy will be tracked
 * individually i.e. as a group with size 1.
 */
TVM_DLL const Op& dma_copy();

/*!
 * \brief Wait until the number of DMA groups in flight is less than
 * or equal to some maximum
 *
 * Calling `dma_wait()` while a group is active is unsupported.
 */
TVM_DLL const Op& dma_wait();

/*!
 * \brief Start a group of DMA copies
 *
 * Any call to `dma_copy()` that occurs after `dma_start_group()` will
 * be added to the current group for tracking of in-flight group counts.
 *
 * Only one DMA group may be active at a given time.  Calling
 * `dma_start_group()` while a group is active is unsupported.
 */
TVM_DLL const Op& dma_start_group();

/*!
 * \brief End a group of DMA copies
 *
 * Track all calls to `dma_copy()` that occurred since the preceding
 * `dma_start_group()` as a single group in-flight.
 *
 * Calling `dma_end_group()` without an active group is unsupported.
 *
 * Note: A group of DMA calls may be empty, and will still contribute
 * to the count of in-flight groups used by `dma_wait()`.
 */
TVM_DLL const Op& dma_end_group();

/*!
 * \brief Provide a true statement that can be used for simplifications
 *
 * Compile-time representation of known constraints about function
 * inputs.  This assumption is removed when lowering, and does not
 * occur in codegen.
 */
TVM_DLL const Op& assume();

/*!
 * \brief Returns an initialized but arbitrary value
 *
 * Compile-time representation of memory locations whose values may be
 * altered as a result of optimizations.
 */
TVM_DLL const Op& undef();

/*!
 * \brief Profiling intrinsic
 */
TVM_DLL const Op& start_profile_intrinsic();

/*!
 * \brief Profiling intrinsic
 */
TVM_DLL const Op& end_profile_intrinsic();

/*!
 * \brief Get a item from any list and return it.
 *
 *  Any anylist_getitem(Handle anylist,
 *                      int index)
 *     return anylist[index];
 *  }
 *
 * \note This intrinsic is only applicable when appearing
 *       in call_packed and anylist_setitem_call_packed.
 */
TVM_DLL const Op& anylist_getitem();

/*!
 * \brief Reset and clear a item in any list.
 *
 *  void anylist_resetitem(Handle anylist,
 *                         int index)
 *    anylist[index] = nullptr;
 *  }
 *
 * \note This intrinsic is only applicable when appearing
 *       in call_packed and anylist_setitem_call_packed.
 */
TVM_DLL const Op& anylist_resetitem();

/*!
 * \brief Set an item into any list by running packed function call.
 *
 *  void anylist_setitem_call_packed(Handle anylist,
 *                                   int index,
 *                                   name, *args)
 *
 *    anylist[index] = call_packed(name, *args)
 *  }
 *  \note This intrinsic can be used in combination with anylist_getitem.
 */
TVM_DLL const Op& anylist_setitem_call_packed();

/*!
 * \brief Same as anylist_setitem_call_packed but use C calling convention.
 */
TVM_DLL const Op& anylist_setitem_call_cpacked();

/*!
 * \brief Get the target's vscale value. It will be lowered to llvm.vscale intrinsic
 * (https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic)
 */
TVM_DLL const Op& vscale();

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

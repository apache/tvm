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
 * \file tvm/tirx/builtin.h
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
#include <tvm/tirx/expr.h>

namespace tvm {
namespace tirx {

/*! \brief Collection of builtin intrinsics as ops */
namespace builtin {
/*!
 * \brief Return value.
 */
TVM_DLL const Op& ret();
/*!
 * \brief Return from a GPU thread.
 */
TVM_DLL const Op& thread_return();
/*!
 * \brief Loop continue.
 */
TVM_DLL const Op& continue_loop();
/*!
 * \brief Loop break.
 */
TVM_DLL const Op& break_loop();
/*!
 * \brief Reinterpret the value using the target type.
 */
TVM_DLL const Op& reinterpret();

/*!
 * \brief Marks a condition is likely going to happen.
 */
TVM_DLL const Op& likely();

/*!
 * \brief Thread-set filter predicate. Used as the condition of an IfThenElse
 * to narrow the active thread set A for the then-branch. Two forms:
 *   filter(var, lo, hi)   -- range form, true iff var in [lo, hi)
 *   filter(var, cond)     -- predicate form (e.g. var == k); true iff cond
 * `var` must be a ScopeIdDef-declared Var at parse time (Verifier Rule 2).
 */
TVM_DLL const Op& filter();

/*!
 * \brief Analysis-only active-thread selector.
 *
 * ``selector(var, pred)`` denotes the unique value of ``var`` in the current
 * active domain for which ``pred`` is true. It is used only inside
 * ExecContext/DispatchContext metadata, for predicates such as
 * ``ptx.elect_sync()`` whose selected lane cannot be inferred structurally.
 */
TVM_DLL const Op& selector();

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
 * \brief same signature as llvm.prefetch
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
 *  void* handle_add_byte_offset(void* handle, int offset) {
 *     return reinterpret_cast<v*>(reinterpret_cast<char*>(handle) + offset);
 *  }
 */
TVM_DLL const Op& handle_add_byte_offset();

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
 * Type lookup_param(ffi::String param_name) {
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
 * \brief Allocate a Tensor(DLTensor) on stack, return the handle.
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
 *  return_type tvm_call_packed(name, TVMFFIAny* args) {
 *     TVMFFIAny result;
 *     ModuleNode* env = GetCurrentEnv();
 *     const ffi::Function* f = env->GetFuncFromEnv(name);
 *     (*f)(args, args, len(args), &result);
 *     // return type can be int, float, handle.
 *     return cast(return_type, result);
 *  }
 */
TVM_DLL const Op& tvm_call_packed();

/*!
 * \brief See pesudo code
 *
 * return_type tvm_call_packed(fname, TVMFFIAny* args) {
 *     TVMFFIAny result;
 *     (*fname)(args, args, len(args), &result);
 *     return cast(return_type, result);
 *  }
 */
TVM_DLL const Op& tvm_call_cpacked();

/*!
 * \brief See pesudo code
 *
 *  return_type tvm_call_trace_packed(name, TVMFFIAny* args) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const ffi::Function* f = env->GetFuncFromEnv(name);
 *     (*f)(args, args, len(args));
 *     // return type can be int, float, handle.
 *     return cast(return_type, result);
 *  }
 */
TVM_DLL const Op& tvm_call_trace_packed();

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
 *                                      TVMFFIAny* args_stack,
 *                                      int begin,
 *                                      int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const ffi::Function* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(ffi::PackedArgs(args_stack[begin:end]),
 *                   ffi::Any(args_stack + end));
 *     // return type can be int, float, handle.
 *     return cast(return_type, load_return_from(args_stack + end))
 *  }
 */
TVM_DLL const Op& tvm_call_packed_lowered();

/*!
 * \brief Lowered version of call c-packed, the space of value and
 *  type codes are explicitly allocated.
 *
 *  int tvm_call_packed_lowered(fname,
 *                              TVMFFIAny* args_stack,
 *                              int begin,
 *                              int end,
 *                              void* self) {
 *     fname(ffi::PackedArgs(value_stack[begin:end], tcode_stack[begin:end]),
 *                   ffi::Any(value_stack + end, tcode_stack + end));
 *  }
 */
TVM_DLL const Op& tvm_call_cpacked_lowered();

/*!
 * \brief Lowered version of trace intrinsic, the space of value and
 *  type codes are explicitly allocated. The return value is the
 *  (end - 1) value on the stack.
 *
 *  return_type tvm_call_trace_packed_lowered(name,
 *                                            TVMFFIAny* args_stack,
 *                                            int begin,
 *                                            int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const ffi::Function* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(ffi::PackedArgs(args_stack[begin:end]),
 *                   ffi::Any(args_stack + end));
 *     // return type can be int, float, handle.
 *     return cast(return_type, load_return_from(args_stack + end))
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
 *  __shfl_down_sync, __shfl_xor_sync and __activemask.
 *
 *  Parameter warp_size is the size of a warp, which helps a backend
 *  to determine whether the width parameter is legal.
 *
 */
TVM_DLL const Op& tvm_warp_shuffle();
TVM_DLL const Op& tvm_warp_shuffle_up();
TVM_DLL const Op& tvm_warp_shuffle_down();
TVM_DLL const Op& tvm_warp_shuffle_xor();
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

// Metal SimdGroup matrix intrinsics

/*!
 * \brief tvm intrinsic for initializing and simdgroup with given value.
 * \note only 8x8 shape is supported by Metal Spec and TVM, but we still keep shape as params,
 *       keeping the similar interface with Metal Spec.
 *
 * void make_filled_simdgroup_matrix(Var d, PrimExpr index, PrimExpr value,
 *                                   int col = 8, int row = 8);
 */
TVM_DLL const Op& make_filled_simdgroup_matrix();

/*!
 * \brief tvm intrinsic for loading data from device memory or threadgroup memory to simdgroup.
 * \note only 8x8 shape is supported by Metal Spec and TVM, but we still keep shape as params,
 *       keeping the similar interface with Metal Spec.
 *
 * void simdgroup_load(Var d, PrimExpr index, PrimExpr ptr, PrimExpr stride,
                       int col = 8, int row = 8, bool transpose_matrix = false);
 */
TVM_DLL const Op& simdgroup_load();

/*!
 * \brief tvm intrinsic for storing data from simdgroup to device memory or threadgroup memory.
 * \note only 8x8 shape is supported by Metal Spec and TVM, but we still keep shape as params,
 *       keeping the similar interface with Metal Spec.
 *
 * void simdgroup_store(Var d, PrimExpr index, PrimExpr ptr, PrimExpr stride,
 *                      int col = 8, int row = 8, bool transpose_matrix = false);
 */
TVM_DLL const Op& simdgroup_store();

/*!
 * \brief tvm intrinsic for multiply and accumulate two matrices in simdgroup
 * \note only 8x8 shape is supported by Metal Spec and TVM, but we still keep shape as params,
 *       keeping the similar interface with Metal Spec.
 *
 * void simdgroup_mma(Var d, PrimExpr index_d, Var a, PrimExpr index_a,
 *                    Var b, PrimExpr index_b, Var c, PrimExpr index_c);
 */
TVM_DLL const Op& simdgroup_multiply_accumulate();

// Metal cooperative_tensor intrinsics (MetalPerformancePrimitives / Metal 4)

/*!
 * \brief Fill a cooperative_tensor with a given value.
 *
 * void cooperative_tensor_fill(Var d, PrimExpr index, PrimExpr value,
 *                              int rows, int cols);
 */
TVM_DLL const Op& cooperative_tensor_fill();

/*!
 * \brief Load data from device or threadgroup memory into a cooperative_tensor.
 *
 * void cooperative_tensor_load(Var d, PrimExpr index, PrimExpr ptr,
 *                              PrimExpr stride, int rows, int cols,
 *                              bool transpose_matrix,
 *                              int mma_M, int mma_N, int mma_K,
 *                              int operand_role);
 * operand_role: 0=left(A), 1=right(B), 2=destination(C)
 */
TVM_DLL const Op& cooperative_tensor_load();

/*!
 * \brief Store data from a cooperative_tensor to device or threadgroup memory.
 *
 * void cooperative_tensor_store(Var d, PrimExpr index, PrimExpr ptr,
 *                               PrimExpr stride, int rows, int cols,
 *                               bool transpose_matrix,
 *                               int mma_M, int mma_N, int mma_K,
 *                               int operand_role);
 * operand_role: 0=left(A), 1=right(B), 2=destination(C)
 */
TVM_DLL const Op& cooperative_tensor_store();

/*!
 * \brief Multiply and accumulate two matrices using cooperative_tensor
 *        (MetalPerformancePrimitives matmul2d).
 *
 * void cooperative_tensor_multiply_accumulate(
 *     Var d, PrimExpr index_d, Var a, PrimExpr index_a,
 *     Var b, PrimExpr index_b, Var c, PrimExpr index_c,
 *     int M, int N, int K, bool transpose_a, bool transpose_b);
 */
TVM_DLL const Op& cooperative_tensor_multiply_accumulate();

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
 * \brief Dot product of two int8x4 vectors and add an optional accumulator
 */
TVM_DLL const Op& dp4a();

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

/*!
 * \brief Calculate a predicate mask given an upper bound (limit) and a current value (base).
 *
 * It will be lowered to the llvm.get.active.lane.mask intrinsic.
 * (https://llvm.org/docs/LangRef.html#llvm-get-active-lane-mask-intrinsics)
 */
TVM_DLL const Op& get_active_lane_mask();

/*! \brief Annotate a predicate not be considered as target condition of loop partition. */
TVM_DLL const Op& ignore_loop_partition();
/*!
 * \brief Get the element offset of a buffer given logical indices.

  The offset is determined by the layout of the buffer.
 */
TVM_DLL const Op& buffer_offset();

/*! \brief The kind of structure field info used in intrinsic */
enum TVMStructFieldKind : int {
  // DLTensor fields
  kDLTensorAddr,
  kDLTensorData,
  kDLTensorShape,
  kDLTensorStrides,
  kDLTensorNDim,
  kDLTensorTypeCode,
  kDLTensorTypeBits,
  kDLTensorTypeLanes,
  kDLTensorByteOffset,
  kDLTensorDeviceId,
  kDLTensorDeviceType,
  kDLTensorKindBound_,
  // TVMValue field
  kTVMValueContent,
  kTVMFFIAnyTypeIndex,
  kTVMFFIAnyZeroPadding,
  kTVMFFIAnyUnionValue,
  kTVMValueKindBound_,
  // Generic int64 array element access: ((int64_t*)buf)[index]
  kInt64ArrayElem,
};

/*!
 * \brief Print the content of a buffer during runtime.
 */
TVM_DLL const Op& print_buffer();

/*!
 * \brief tvm intrinsic for initializing the CUDA profiler, and store profiling result in a buffer.
 *
 *  void timer_init_cuda(Var profiler_buffer, Var profiler_tag, Var profiler_write_offset, int
 * num_groups, Expr group_id) {
 *    // initialize the tag and write to pos 0 in the buffer
 *    // initialize write offset for every leader thread in warp group across all blocks
 *  }
 */
TVM_DLL const Op& timer_init_cuda();

/*!
 * \brief tvm intrinsic for starting the timer for profiling a specific event,
 *        and storing profiling result in a buffer.
 *
 *  void timer_start_cuda(IntImm event_type, Var profiler_buffer, Var profiler_tag,
 *                        Var profiler_write_offset, IntImm profiler_write_stride, Expr leader_cond)
 * {
 *    // each leader thread in warp group gets the time stamp and event type, combine with the tag
 *    // and write to corresponding offset in buffer
 *    // each leader thread advance offset by stride
 *  }
 */
TVM_DLL const Op& timer_start_cuda();

/*!
 * \brief tvm intrinsic for ending the timer for profiling a specific event,
 *        and storing profiling result in a buffer.
 *
 *  void timer_end_cuda(IntImm event_type, Var profiler_buffer, Var profiler_tag,
 *                      Var profiler_write_offset, IntImm profiler_write_stride, Expr leader_cond) {
 *    // each leader thread in warp group gets the time stamp and event type, combine with the tag
 *    // and write to corresponding offset in buffer
 *    // each leader thread advance offset by stride
 *  }
 */
TVM_DLL const Op& timer_end_cuda();

/*!
 * \brief tvm intrinsic for finalize the timer for profiling,
 *        and storing profiling result in a buffer.
 *
 *  void timer_finalize_cuda(Var profiler_buffer, Var profiler_tag, Var profiler_write_offset,
 *                          IntImm profiler_write_stride, Expr leader_cond) {
 *    // each leader thread in warp group gets the time stamp and end signal, combine with the tag
 *    // and write to corresponding offset in buffer
 *    // each leader thread advance offset by stride
 *  }
 */
TVM_DLL const Op& timer_finalize_cuda();

/*!
 * \brief tvm intrinsic for cuda atomic add instruction
 */
TVM_DLL const Op& cuda_atomic_add();

/*!
 * \brief tvm intrinsic for cuda thread fence instruction
 */
TVM_DLL const Op& cuda_thread_fence();

/*!
 * \brief Warp-level butterfly shuffle-XOR reduction.
 *
 * cuda_warp_reduce(value, op, width) reduces value across width adjacent
 * lanes using the specified operation ("sum", "max", "min").
 */
TVM_DLL const Op& cuda_warp_reduce();

/*!
 * \brief CTA-wide reduction via warp shuffle + shared memory.
 *
 * cuda_cta_reduce(value, op, num_warps, scratch) reduces value across
 * the entire CTA using the specified operation ("sum", "max", "min").
 */
TVM_DLL const Op& cuda_cta_reduce();

/*!
 * \brief Typed load/store copy of num_bytes bytes.
 *
 * cuda_copy_bytes(dst, src, num_bytes) copies num_bytes bytes from src to dst
 * using a single typed load/store (uint4, uint2, unsigned int, etc.).
 * num_bytes must be one of {1, 2, 4, 8, 16}.
 */
TVM_DLL const Op& cuda_copy_bytes();

/*!
 * \brief tvm intrinsic for cuda warp sync instruction
 */
TVM_DLL const Op& cuda_warp_sync();

/*!
 * \brief tvm intrinsic for cuda block-wide sync (syncthreads)
 */
TVM_DLL const Op& cuda_cta_sync();

/*!
 * \brief tvm intrinsic for cuda grid-wide sync (cooperative groups)
 */
TVM_DLL const Op& cuda_grid_sync();

/*!
 * \brief tvm intrinsic that returns ``cooperative_groups::thread_rank()``
 *        for the enclosing CTA (linear thread index within the block).
 */
TVM_DLL const Op& cuda_thread_rank();

/*!
 * \brief tvm intrinsic for cuda half to float conversion
 */
TVM_DLL const Op& cuda_half2float();

/*!
 * \brief tvm intrinsic for cuda bfloat16 to float conversion
 */
TVM_DLL const Op& cuda_bfloat162float();

/*!
 * \brief tvm intrinsic for a helper converting float2 to half2 with rounding
 */
TVM_DLL const Op& cuda_float22half2();

/*!
 * \brief tvm intrinsic to trap when an assertion failed (cond == false)
 */
TVM_DLL const Op& cuda_trap_when_assert_failed();

/*!
 * \brief tvm intrinsic to modify runtime instruction descriptor
 */
TVM_DLL const Op& cuda_runtime_instr_desc();

/*!
 * \brief tvm intrinsic to convert 8 half2 lanes to 8 float2 lanes
 */
TVM_DLL const Op& cuda_half8tofloat8();

/*!
 * \brief tvm intrinsic to convert 8 float2 lanes to 8 half2 lanes with rounding
 */
TVM_DLL const Op& cuda_float8tohalf8();

/*!
 * \brief tvm intrinsic for cuda syncthreads_and instruction
 */
TVM_DLL const Op& cuda_syncthreads_and();

/*!
 * \brief tvm intrinsic for cuda syncthreads_or instruction
 */
TVM_DLL const Op& cuda_syncthreads_or();

/*!
 * \brief tvm intrinsic for cuda nano sleep instruction
 */
TVM_DLL const Op& cuda_nano_sleep();

/*!
 * \brief tvm intrinsic for cuda atomic compare and swap instruction
 */
TVM_DLL const Op& cuda_atomic_cas();

/*!
 * \brief tvm intrinsic for cuda printf instruction
 */
TVM_DLL const Op& cuda_printf();

/*!
 * \brief tvm intrinsic for cuda ldg instruction
 */
TVM_DLL const Op& cuda_ldg();

/*!
 * \brief tvm intrinsic for cuda tmem address calculation
 */
TVM_DLL const Op& cuda_get_tmem_addr();

/*!
 * \brief tvm intrinsic for PTX fast exp2 approximation (ex2.approx.ftz.f32)
 */
TVM_DLL const Op& ptx_exp2();

/*!
 * \brief tvm intrinsic for PTX fast reciprocal approximation (rcp.approx.ftz.f32)
 */
TVM_DLL const Op& ptx_rcp();

/*!
 * \brief tvm intrinsic for PTX warp-wide any predicate (__any_sync)
 */
TVM_DLL const Op& ptx_any_sync();

/*!
 * \brief tvm intrinsic for PTX 3-input max instruction (sm_100a+)
 */
TVM_DLL const Op& ptx_reduce3_max_f32();

/*!
 * \brief tvm intrinsic for PTX 3-input min instruction (sm_100a+)
 */
TVM_DLL const Op& ptx_reduce3_min_f32();

/*!
 * \brief tvm intrinsic for PTX packed add instruction (sm_100a+)
 */
TVM_DLL const Op& ptx_add_packed_f32x2();

/*!
 * \brief tvm intrinsic for PTX packed subtract instruction (sm_100a+)
 */
TVM_DLL const Op& ptx_sub_packed_f32x2();

/*!
 * \brief tvm intrinsic for PTX packed multiply instruction (sm_100a+)
 */
TVM_DLL const Op& ptx_mul_packed_f32x2();

/*!
 * \brief tvm intrinsic for PTX packed FMA instruction (sm_100a+)
 */
TVM_DLL const Op& ptx_fma_packed_f32x2();

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_BUILTIN_H_

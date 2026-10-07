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
 * \brief TIRx builtin region and call operations.
 *
 * TIRx builtin operations are stored as tvm::Op.
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
#include <tvm/ir/prim/expr.h>

namespace tvm {
namespace tirx {

/*! \brief Collection of builtin region and call operations as Ops. */
namespace builtin {
/*!
 * \name Region operations
 * \brief Operations used by RegionStmt to enclose a lexical body.
 *
 * Every region operation registers FRegionGetBodyParams, returning fresh typed
 * body parameters or an empty array for no parameters. Attribute presence alone
 * identifies region support; classification does not invoke the hook. Operands
 * and attributes are evaluated outside the body-parameter scope.
 * \{
 */
/*!
 * \brief Thread launch region: operands are a nonempty StringImm tag and a
 * scalar signed or unsigned integer extent wider than one bit.
 * The sole body parameter is a fresh thread-index PrimVar matching the extent type.
 * Tags starting with vthread denote virtual threads. There are no attrs or results.
 */
TVM_DLL const Op& launch_thread();

/*!
 * \brief Mark a user-facing device entry containing device scope definitions.
 * Takes no operands, body parameters, attributes or results.
 */
TVM_DLL const Op& device_entry();
/*!
 * \brief Supply lexical device context for allocation and packed-call lowering.
 * Operands are integer device type and device ID; there are no body parameters,
 * attributes or results. The region does not change the active runtime device.
 */
TVM_DLL const Op& device_context();
/*!
 * \brief Outline the body as a CPU compute helper named by a StringImm operand.
 * There are no body parameters, attributes or results.
 */
TVM_DLL const Op& compute_scope();
/*!
 * \brief Launch a CPU worker team around parallel loops and team barriers.
 * Takes no operands, body parameters, attributes or results.
 */
TVM_DLL const Op& parallel_launch();
/*! \} */

/*!
 * \name Call operations
 * \brief Operations invoked through Call with explicit operands and result types.
 * \{
 */
/*!
 * \brief Allocate a buffer: alloc_tensor(shape, dtype, scope) -> TensorType.
 *
 * Arguments, in order:
 * - args[0]: shape, Tuple of integer extents (IntImm or symbolic integer expressions).
 * - args[1]: dtype, DataTypeImm with a DLDataType payload for the element type.
 * - args[2]: scope, StringImm naming the storage scope.
 *
 * DictAttrs directly holds the allocation annotations, defaulting to an empty dictionary.
 * The TensorType result agrees with the operands and retains buffer access/storage metadata.
 *
 * \code
 * // Example pattern match code for a given Binding:
 * if (const auto* call = binding->value.as<CallNode>();
 *     call && call->op.same_as(builtin::alloc_tensor())) {
 *   tvm::Tuple shape = call->args[0].as_or_throw<tvm::Tuple>();
 *   DLDataType dtype = call->args[1].as_or_throw<DataTypeImm>()->value;
 *   ffi::String scope = call->args[2].as_or_throw<StringImm>()->value;
 *   DictAttrs annotations = call->attrs.as_or_throw<DictAttrs>();
 * }
 * \endcode
 */
TVM_DLL const Op& alloc_tensor();
/*!
 * \brief Declare a buffer view: decl_tensor(data, shape, dtype, scope) -> TensorType.
 *
 * Arguments, in order:
 * - args[0]: data, Expr for the existing physical pointer backing the buffer view.
 * - args[1]: shape, Tuple of integer extents (IntImm or symbolic integer expressions).
 * - args[2]: dtype, DataTypeImm with a DLDataType payload for the element type.
 * - args[3]: scope, StringImm naming the storage scope.
 *
 * There are no attributes. The TensorType result agrees with the operands and retains
 * buffer access/storage metadata. The operation binds a view without allocating memory.
 *
 * \code
 * // Example pattern match code for a given Binding:
 * if (const auto* call = binding->value.as<CallNode>();
 *     call && call->op.same_as(builtin::decl_tensor())) {
 *   Expr data = call->args[0];
 *   tvm::Tuple shape = call->args[1].as_or_throw<tvm::Tuple>();
 *   DLDataType dtype = call->args[2].as_or_throw<DataTypeImm>()->value;
 *   ffi::String scope = call->args[3].as_or_throw<StringImm>()->value;
 * }
 * \endcode
 */
TVM_DLL const Op& decl_tensor();
/*!
 * \brief Return from a GPU thread without returning a function value.
 */
TVM_DLL const Op& thread_return();
/*!
 * \brief Reinterpret the value using the target type.
 */
TVM_DLL const Op& reinterpret();

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
 * \brief Execute a multiplication between two Q-numbers x and y
 * followed by a right shift s
 * The default rounding rule is to the nearest value, rounding half up
 * (i.e., round(x.1) = x and round (x.5) = x+1)
 */
TVM_DLL const Op& q_multiply_shift();
TVM_DLL const Op& q_multiply_shift_per_axis();

/*!
 * \brief Returns the address of an element in the buffer (see pseudocode below).
 *
 * The number of indices should match the dimensionality of the buffer
 * being accessed.  If this operation occurs after buffer flattening,
 * the number of indices must be supported by the target (i.e. N>1
 * only on targets that support non-flat memory buffers).
 *
 *  Handle address_of(TensorLoad *op) {
 *     return &op->buffer_var[op->indices[0], op->indices[1], ..., op->indices[N-1]];
 *  }
 */
TVM_DLL const Op& address_of();

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
 * \brief Cast a handle to a typed pointer after adding a byte offset.
 *
 *  DType* ptr_byte_offset(void* data, int byte_offset, Expr dtype) {
 *    return reinterpret_cast<DType*>(reinterpret_cast<char*>(data) + byte_offset);
 *  }
 */
TVM_DLL const Op& ptr_byte_offset();

/*!
 * \brief Create a function local static handle that iniitalizes to nullptr.
 *  can be used to cache function local static resources.
 */
TVM_DLL const Op& tvm_static_handle();

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
 * \brief Launch a kernel using the packed-function argument convention.
 *
 * Arguments are the kernel symbol, kernel operands, then launch values.
 * CallFFIKernelAttr::launch_params describes the launch-value suffix.
 * Host backends may consume this call directly or lower it to tvm_call_packed.
 */
TVM_DLL const Op& call_ffi_kernel();

/*!
 * \brief Encode a tiled tensor map at invocation time.
 *
 * TensorMapEncodeTiledAttr stores the descriptor dtype, rank and fixed options.
 * Arguments are descriptor and data pointers, global dimensions (rank), byte
 * strides (rank - 1), box dimensions (rank), then element strides (rank).
 */
TVM_DLL const Op& tensormap_encode_tiled();

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
 * \brief See pseudo code
 *
 *  int tvm_storage_sync(std::string storage_scope) {
 *     __sync(storage_scope);
 *     return 0;
 *  }
 */
TVM_DLL const Op& tvm_storage_sync();

/*!
 * \brief Synchronize all workers in the current CPU parallel launch.
 *
 * Every worker must reach this operation. It must be outside partitioned
 * parallel loops, whose iteration counts can differ between workers.
 */
TVM_DLL const Op& cpu_parallel_barrier();

/*!
 * \brief Marker where a transform should replace generated kernel initialization.
 */
TVM_DLL const Op& tvm_kernel_replace_point();

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
 * \brief Cross-thread reduction with an explicit typed combiner and identities.
 *
 * void tvm_thread_allreduce(LambdaExpr combine, Expr identity, Expr values,
 *                           PrimExpr predicate, Expr destinations, Expr thread_axes);
 *
 * For N values, combine binds lhs[0:N] followed by rhs[0:N] and returns an
 * N-element Tuple, or a scalar when N is one. Identity, values, destinations
 * and thread_axes may each be a scalar or an explicit Tuple of fields.
 * Each value, identity, pair of parameters and result have
 * the same primitive type. Inactive inputs are replaced by their identities.
 * Destinations are N tensor loads at index zero of one-element result temporaries
 * (optionally cast for boolean storage). Each result temporary must be accessed
 * only at index zero. Thread axes are reduction thread variables or zero for
 * simplified unit axes.
 * Other thread indices remain fixed. The operation writes the reduced values
 * to the destination tensors and returns void.
 */
TVM_DLL const Op& tvm_thread_allreduce();

/*!
 * \brief View a scalar all-reduce operand/result as one field, or expose its Tuple fields.
 * \param value The scalar expression or explicit Tuple.
 * \return The fields without changing the expression's representation in IR.
 */
inline ffi::Array<Expr> GetAllreduceFields(const Expr& value) {
  if (const auto* tuple = value.as<tvm::TupleNode>()) return tuple->fields;
  return {value};
}

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
 * \brief Provide a true statement that can be used for simplifications
 *
 * Compile-time representation of known constraints about function
 * inputs.  This assumption is removed when lowering, and does not
 * occur in codegen.
 */
TVM_DLL const Op& assume();

/*!
 * \brief Assume a tensor's base address has the given constant byte alignment.
 *
 * This leaf operation carries a compiler fact, without checking or changing
 * the address. Alignment must be a power of two between 1 and 2^27 bytes.
 */
TVM_DLL const Op& assume_aligned();

/*!
 * \brief Returns an initialized but arbitrary value
 *
 * Compile-time representation of memory locations whose values may be
 * altered as a result of optimizations.
 */
TVM_DLL const Op& undef();

/*!
 * \brief Calculate a predicate mask given an upper bound (limit) and a current value (base).
 *
 * It will be lowered to the llvm.get.active.lane.mask intrinsic.
 * (https://llvm.org/docs/LangRef.html#llvm-get-active-lane-mask-intrinsics)
 */
TVM_DLL const Op& get_active_lane_mask();

/*!
 * \brief Masked buffer load.
 *
 * Arguments are the buffer variable, one or more indices, and a trailing boolean lane mask.
 * The result type is the vector type loaded from the selected lanes.
 */
TVM_DLL const Op& masked_load();

/*!
 * \brief Masked buffer store.
 *
 * Arguments are the buffer variable, value, one or more indices, and a trailing boolean lane
 * mask. The result type is void.
 */
TVM_DLL const Op& masked_store();

/*! \brief Annotate a predicate not be considered as target condition of loop partition. */
TVM_DLL const Op& ignore_loop_partition();
/*!
 * \brief Get the element offset of a buffer given logical indices.

  The offset is determined by the layout of the buffer.
 */
TVM_DLL const Op& buffer_offset();

/*!
 * \brief Project the physical pointer associated with a TensorVar definition.
 *
 * The result pointer type is derived from the TensorType dtype and storage
 * scope of the sole TensorVar argument.  This operation is consumed by TIRx
 * lowering and code generation.
 */
TVM_DLL const Op& buffer_data();

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
/*! \} */
}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_BUILTIN_H_

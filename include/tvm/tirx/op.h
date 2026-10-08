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
 * \file tvm/tirx/op.h
 * \brief TIRx operator attributes, builtin Ops and expression helpers.
 *
 * \note Most of the operator defined here perform simple constant folding
 *   when the type is int32 or int64 for simplifying the index expressions.
 */
// Acknowledgement: Most operator APIs originate from Halide.
#ifndef TVM_TIRX_OP_H_
#define TVM_TIRX_OP_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

namespace tvm {
namespace tirx {

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
 *     call && call->op.same_as(tirx::alloc_tensor_op())) {
 *   tvm::Tuple shape = call->args[0].as_or_throw<tvm::Tuple>();
 *   DLDataType dtype = call->args[1].as_or_throw<DataTypeImm>()->value;
 *   ffi::String scope = call->args[2].as_or_throw<StringImm>()->value;
 *   DictAttrs annotations = call->attrs.as_or_throw<DictAttrs>();
 * }
 * \endcode
 */
TVM_DLL const Op& alloc_tensor_op();
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
 *     call && call->op.same_as(tirx::decl_tensor_op())) {
 *   Expr data = call->args[0];
 *   tvm::Tuple shape = call->args[1].as_or_throw<tvm::Tuple>();
 *   DLDataType dtype = call->args[2].as_or_throw<DataTypeImm>()->value;
 *   ffi::String scope = call->args[3].as_or_throw<StringImm>()->value;
 * }
 * \endcode
 */
TVM_DLL const Op& decl_tensor_op();
/*!
 * \brief Return from a GPU thread without returning a function value.
 */
TVM_DLL const Op& thread_return_op();
/*!
 * \brief Reinterpret the value using the target type.
 */
TVM_DLL const Op& reinterpret_op();

/*!
 * \brief Thread-set filter predicate. Used as the condition of an IfThenElse
 * to narrow the active thread set A for the then-branch. Two forms:
 *   filter(var, lo, hi)   -- range form, true iff var in [lo, hi)
 *   filter(var, cond)     -- predicate form (e.g. var == k); true iff cond
 * `var` must be a ScopeIdDef-declared Var at parse time (Verifier Rule 2).
 */
TVM_DLL const Op& filter_op();

/*!
 * \brief Analysis-only active-thread selector.
 *
 * ``selector(var, pred)`` denotes the unique value of ``var`` in the current
 * active domain for which ``pred`` is true. It is used only inside
 * ExecContext/DispatchContext metadata, for predicates such as
 * ``ptx.elect_sync()`` whose selected lane cannot be inferred structurally.
 */
TVM_DLL const Op& selector_op();

/*!
 * \brief Execute a multiplication between two Q-numbers x and y
 * followed by a right shift s
 * The default rounding rule is to the nearest value, rounding half up
 * (i.e., round(x.1) = x and round (x.5) = x+1)
 */
TVM_DLL const Op& q_multiply_shift_op();
TVM_DLL const Op& q_multiply_shift_per_axis_op();

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
TVM_DLL const Op& address_of_op();

/*!
 * \brief See pesudo code
 *
 *  bool isnullptr(void* handle) {
 *     return handle == nullptr
 *  }
 */
TVM_DLL const Op& isnullptr_op();

/*!
 * \brief Check if value is nan
 */
TVM_DLL const Op& isnan_op();

/*!
 * \brief Popcount
 */
TVM_DLL const Op& popcount_op();

/*!
 * \brief Fused multiply add
 *
 *  Type fma(a, b, c) {
 *    return a * b + c;
 *  }
 */
TVM_DLL const Op& fma_op();

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
TVM_DLL const Op& call_extern_op();

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
TVM_DLL const Op& call_pure_extern_op();

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
TVM_DLL const Op& call_llvm_intrin_op();

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
TVM_DLL const Op& call_llvm_pure_intrin_op();

/*!
 * \brief Call an SPIRV pure GLSL450 intrinsic.
 *
 *  Type call_spirv_pure_glsl450(intrin_id, args...) {
 *     return dlsym(name)(args...);
 *  }
 *
 * \note This op does not provide any type checking.
 */
TVM_DLL const Op& call_spirv_pure_glsl450_op();

// TODO(tvm-team) revisit the builtins below
// some of them can simply become ops with special codegen attr.
/*!
 * \brief same signature as llvm.prefetch
 */
TVM_DLL const Op& prefetch_op();

/*!
 * \brief Get head access address with memory access pattern info.
 *
 *  This operator also marks range of the memory access
 *  The offset and extent are in unit of the DType(including vectorization factor).
 *  rw_mask is a bit_mask setting whether the access is a read(1) or write(2).
 *  The access is assume to happen in the current expression.
 *
 *  PtrType tvm_access_ptr<DType>(DType* data,
 *                         int offset, int extent,
 *                         int rw_mask) {
 *    // DType is the independent access type in ty_args[0].
 *    return &data[offset];
 *  }
 */
TVM_DLL const Op& tvm_access_ptr_op();

/*!
 * \brief Cast a handle to a typed pointer after adding a byte offset.
 *
 *  DType* ptr_byte_offset(void* data, int byte_offset) {
 *    return reinterpret_cast<DType*>(reinterpret_cast<char*>(data) + byte_offset);
 *  }
 */
TVM_DLL const Op& ptr_byte_offset_op();

/*!
 * \brief Create a function local static handle that iniitalizes to nullptr.
 *  can be used to cache function local static resources.
 */
TVM_DLL const Op& tvm_static_handle_op();

/*!
 * \brief See pesudo code
 *
 *  void* handle_add_byte_offset(void* handle, int offset) {
 *     return reinterpret_cast<v*>(reinterpret_cast<char*>(handle) + offset);
 *  }
 */
TVM_DLL const Op& handle_add_byte_offset_op();

/*!
 * \brief See pesudo code
 *
 *  Type tvm_struct_get(StructType* arr, int index, int field_id) {
 *     return arr[index]->field;
 *  }
 * \sa TVMStructFieldKind
 */
TVM_DLL const Op& tvm_struct_get_op();

/*!
 * \brief See pesudo code
 *
 *  Handle tvm_struct_set(StructType* arr, int index, int field_id, value) {
 *     arr[index]->field = value;
 *  }
 * \sa TVMStructFieldKind
 */
TVM_DLL const Op& tvm_struct_set_op();

/*!
 * \brief See pesudo code
 *
 *  void tvm_throw_last_error() {
 *    throw TVMGetLastError();
 *  }
 */
TVM_DLL const Op& tvm_throw_last_error_op();

/*!
 * \brief See pesudo code
 *
 *  dtype in {shape, array, arg_value, arg_tcode}
 *
 *  Handle tvm_stack_alloca(string dtype, int num) {
 *     return new on stack dtype[num];
 *  }
 */
TVM_DLL const Op& tvm_stack_alloca_op();

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
TVM_DLL const Op& tvm_stack_make_shape_op();

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
TVM_DLL const Op& tvm_stack_make_array_op();

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
TVM_DLL const Op& tvm_call_packed_op();

/*! \brief Launch metadata for call_ffi_kernel. */
struct CallFFIKernelAttr : public AttrsNode {
  /*!
   * \brief Ordered launch tags describing the suffix of the call arguments.
   *
   * The first call argument is the kernel symbol, followed by kernel operands
   * and launch values. Flag-only tags consume no value; dynamic shared-memory
   * bytes, when present, are last. Runtime expressions remain in Call.args.
   */
  ffi::Array<ffi::String> launch_params;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.CallFFIKernelAttr", CallFFIKernelAttr, AttrsNode);
};

/*!
 * \brief Launch a kernel using the packed-function argument convention.
 *
 * Arguments are the kernel symbol, kernel operands, then launch values.
 * CallFFIKernelAttr::launch_params describes the launch-value suffix.
 * Host backends may consume this call directly or lower it to tvm_call_packed.
 */
TVM_DLL const Op& call_ffi_kernel_op();

/*! \brief Fixed encoding options for tensormap_encode_tiled. */
struct TensorMapEncodeTiledAttr : public AttrsNode {
  DLDataType descriptor_dtype;
  int64_t rank;
  int64_t interleave;
  int64_t swizzle;
  int64_t l2_promotion;
  int64_t oob_fill;
  int64_t force_cu_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TensorMapEncodeTiledAttr", TensorMapEncodeTiledAttr,
                                    AttrsNode);
};

/*!
 * \brief Encode a tiled tensor map at invocation time.
 *
 * TensorMapEncodeTiledAttr stores the descriptor dtype, rank and fixed options.
 * Arguments are descriptor and data pointers, global dimensions (rank), byte
 * strides (rank - 1), box dimensions (rank), then element strides (rank).
 */
TVM_DLL const Op& tensormap_encode_tiled_op();

/*!
 * \brief See pesudo code
 *
 * return_type tvm_call_packed(fname, TVMFFIAny* args) {
 *     TVMFFIAny result;
 *     (*fname)(args, args, len(args), &result);
 *     return cast(return_type, result);
 *  }
 */
TVM_DLL const Op& tvm_call_cpacked_op();

/*!
 * \brief Mark a condition to be thread invariant.
 *  This means the condition must be the same for all threads.
 */
TVM_DLL const Op& tvm_thread_invariant_op();

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
TVM_DLL const Op& tvm_call_packed_lowered_op();

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
TVM_DLL const Op& tvm_call_cpacked_lowered_op();

/*!
 * \brief See pseudo code
 *
 *  int tvm_storage_sync(std::string storage_scope) {
 *     __sync(storage_scope);
 *     return 0;
 *  }
 */
TVM_DLL const Op& tvm_storage_sync_op();

/*!
 * \brief Synchronize all workers in the current CPU parallel launch.
 *
 * Every worker must reach this operation. It must be outside partitioned
 * parallel loops, whose iteration counts can differ between workers.
 */
TVM_DLL const Op& cpu_parallel_barrier_op();

/*!
 * \brief Marker where a transform should replace generated kernel initialization.
 */
TVM_DLL const Op& tvm_kernel_replace_point_op();

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
TVM_DLL const Op& tvm_warp_shuffle_op();
TVM_DLL const Op& tvm_warp_shuffle_up_op();
TVM_DLL const Op& tvm_warp_shuffle_down_op();
TVM_DLL const Op& tvm_warp_shuffle_xor_op();
TVM_DLL const Op& tvm_warp_activemask_op();

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
TVM_DLL const Op& tvm_thread_allreduce_op();

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
TVM_DLL const Op& cooperative_tensor_fill_op();

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
TVM_DLL const Op& cooperative_tensor_load_op();

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
TVM_DLL const Op& cooperative_tensor_store_op();

/*!
 * \brief Multiply and accumulate two matrices using cooperative_tensor
 *        (MetalPerformancePrimitives matmul2d).
 *
 * void cooperative_tensor_multiply_accumulate(
 *     Var d, PrimExpr index_d, Var a, PrimExpr index_a,
 *     Var b, PrimExpr index_b, Var c, PrimExpr index_c,
 *     int M, int N, int K, bool transpose_a, bool transpose_b);
 */
TVM_DLL const Op& cooperative_tensor_multiply_accumulate_op();

// TODO(tvm-team) replace the usage of the vector operations by Shuffle.
/*!
 * \brief Get the high level half of the vector
 */
TVM_DLL const Op& vectorhigh_op();

/*!
 * \brief Get the low-level half of the vector
 */
TVM_DLL const Op& vectorlow_op();

/*!
 * \brief Concat two vectors.
 */
TVM_DLL const Op& vectorcombine_op();

/*!
 * \brief Dot product of two int8x4 vectors and add an optional accumulator
 */
TVM_DLL const Op& dp4a_op();

/*!
 * \brief atomic add instruction, corresponding e.g. to atomicAdd in CUDA
 */
TVM_DLL const Op& atomic_add_op();
/*!
 * \brief Create an Nd memory allocation with storage scope
 */
TVM_DLL const Op& nd_mem_alloc_with_scope_op();

/*!
 * \brief Store to texture 2d memory
 */
TVM_DLL const Op& texture2d_store_op();

/*!
 * \brief Load from texture 2d memory
 */
TVM_DLL const Op& texture2d_load_op();

/*!
 * \brief Provide a true statement that can be used for simplifications
 *
 * Compile-time representation of known constraints about function
 * inputs.  This assumption is removed when lowering, and does not
 * occur in codegen.
 */
TVM_DLL const Op& assume_op();

/*!
 * \brief Assume a tensor's base address has the given constant byte alignment.
 *
 * This leaf operation carries a compiler fact, without checking or changing
 * the address. Alignment must be a power of two between 1 and 2^27 bytes.
 */
TVM_DLL const Op& assume_aligned_op();

/*!
 * \brief Returns an initialized but arbitrary value
 *
 * Compile-time representation of memory locations whose values may be
 * altered as a result of optimizations.
 */
TVM_DLL const Op& undef_op();

/*!
 * \brief Calculate a predicate mask given an upper bound (limit) and a current value (base).
 *
 * It will be lowered to the llvm.get.active.lane.mask intrinsic.
 * (https://llvm.org/docs/LangRef.html#llvm-get-active-lane-mask-intrinsics)
 */
TVM_DLL const Op& get_active_lane_mask_op();

/*!
 * \brief Masked buffer load.
 *
 * Arguments are the buffer variable, one or more indices, and a trailing boolean lane mask.
 * The result type is the vector type loaded from the selected lanes.
 */
TVM_DLL const Op& masked_load_op();

/*!
 * \brief Masked buffer store.
 *
 * Arguments are the buffer variable, value, one or more indices, and a trailing boolean lane
 * mask. The result type is void.
 */
TVM_DLL const Op& masked_store_op();

/*! \brief Annotate a predicate not be considered as target condition of loop partition. */
TVM_DLL const Op& ignore_loop_partition_op();
/*!
 * \brief Get the element offset of a buffer given logical indices.

  The offset is determined by the layout of the buffer.
 */
TVM_DLL const Op& buffer_offset_op();

/*!
 * \brief Project the physical pointer associated with a TensorVar definition.
 *
 * The result pointer type is derived from the TensorType dtype and storage
 * scope of the sole TensorVar argument.  This operation is consumed by TIRx
 * lowering and code generation.
 */
TVM_DLL const Op& buffer_data_op();

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
TVM_DLL const Op& print_buffer_op();
/*! \} */

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
TVM_DLL const Op& launch_thread_op();

/*!
 * \brief Mark a user-facing device entry containing device scope definitions.
 * Takes no operands, body parameters, attributes or results.
 */
TVM_DLL const Op& device_entry_op();
/*!
 * \brief Supply lexical device context for allocation and packed-call lowering.
 * Operands are integer device type and device ID; there are no body parameters,
 * attributes or results. The region does not change the active runtime device.
 */
TVM_DLL const Op& device_context_op();
/*!
 * \brief Outline the body as a CPU compute helper named by a StringImm operand.
 * There are no body parameters, attributes or results.
 */
TVM_DLL const Op& compute_scope_op();
/*!
 * \brief Launch a CPU worker team around parallel loops and team barriers.
 * Takes no operands, body parameters, attributes or results.
 */
TVM_DLL const Op& parallel_launch_op();
/*! \} */

}  // namespace tirx
}  // namespace tvm

namespace tvm::prim {

// Shared primitive construction and constants are declared in ir/prim/op.h.

/*!
 * \brief Get the type of the expression under the unified type system.
 *
 * This function could return a more refined type than the runtime dtype
 * implied by PrimExpr::ty().
 *
 * \param expr The input parameter.
 * \return The result type.
 *
 * \sa tvm/ir/type.h for discussion about the relation between Type and DLPack dtype.
 */
TVM_DLL Type GetType(const PrimExpr& expr);

/*!
 * \brief Get the type corresponding to a runtime DLPack dtype.
 * \param dtype The runtime dtype.
 * \return The result type
 *
 * \sa tvm/ir/type.h for discussion about the relation between Type and DLPack dtype.
 */
TVM_DLL Type GetTypeFromRuntimeDataType(DLDataType dtype);

/*!
 * \brief Return from a GPU thread without returning a function value.
 *
 * \param span The location of this operation in the source.
 * \return The thread return expression.
 */
TVM_DLL PrimExpr thread_return(Span span = Span());

/*!
 * Get the value of infinity.
 * \param dtype The primitive type.
 * \param span The location of this operation in the source.
 * \return the infinity value in this format.
 */
TVM_DLL PrimExpr infinity(PrimType dtype, Span span = Span());

/*!
 * \brief perform reinterpret cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note This function may return value if the type is the same.
 */
TVM_DLL PrimExpr reinterpret(PrimType t, PrimExpr value, Span span = Span());
TVM_DLL PrimExpr reinterpret(DLDataType t, PrimExpr value, Span span = Span());
/*! \brief Perform a reinterpret cast involving an exact primitive or pointer type. */
TVM_DLL Expr reinterpret(Type target_ty, Expr value, Span span = Span());

/*!
 * \brief Compute log(exp(a) + exp(b)).
 *
 * \param a Left operand.
 * \param b Right operand.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr logaddexp(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief Calculate power(x, y)
 * \param x The left operand.
 * \param y The right operand.
 * \param span The location of this operation in the source.
 */
TVM_DLL PrimExpr pow(PrimExpr x, PrimExpr y, Span span = Span());
/*!
 * \brief Calculate absolute value of x.
 * \param x The input data
 * \param span The location of this operation in the source.
 *
 * \return The absolute value of input data x
 */
TVM_DLL PrimExpr abs(PrimExpr x, Span span = Span());
/*!
 * \brief Check if x is NaN.
 * \param x The input data
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr isnan(PrimExpr x, Span span = Span());

/*!
 * \brief Check if x is finite.
 * \param x The input data
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr isfinite(PrimExpr x, Span span = Span());

/*!
 * \brief Check if x is infinite.
 * \param x The input data
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr isinf(PrimExpr x, Span span = Span());

/*!
 * \brief Calculate floor(x)
 * \param x The input expression.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr floor(PrimExpr x, Span span = Span());

/*!
 * \brief Round x to the nearest integer, ties to even.
 *
 * Uses IEEE 754 default rounding mode (ties-to-even / banker's rounding).
 * Constant-folding and all backends consistently use std::nearbyint semantics.
 *
 * \param x The input expression.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr round(PrimExpr x, Span span = Span());

/*!
 * \brief Round x to the nearest integer, ties to even.
 *
 * Equivalent to round(). Both use IEEE 754 default rounding mode (ties-to-even).
 *
 * \param x The input expression.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr nearbyint(PrimExpr x, Span span = Span());

/*!
 * \brief Calculate trunc(x)
 * \param x The input expression.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr trunc(PrimExpr x, Span span = Span());

/*!
 * \brief Execute a multiplication between two Q-numbers x and y
 * followed by a right shift s. The mathematical expression is:
 *
 *    out = round(x*y*2^-s)
 *
 * Please note that the two Q-numbers x and y are supposed to have
 * the same number of fractional bits q.
 *
 * More about Q-numbers here: https://en.wikipedia.org/wiki/Q_(number_format)
 *
 * The rounding rule is to the nearest value, rounding half up
 * (i.e., round(x.1) = x and round (x.5) = x+1)
 * \param x first Q-number
 * \param y second Q-number
 * \param q number of fractional bits in x and y. Needs to be > 0
 * \param s integer right shift
 * \param span The location of this operation in the source.
 * \return The constructed expression.
 */
TVM_DLL PrimExpr q_multiply_shift(PrimExpr x, PrimExpr y, PrimExpr q, PrimExpr s,
                                  Span span = Span());

/*!
 * \brief Fast_erf_float expression from Eigen
 *
 * \param arg The input expression.
 * \param bits The number of bits in the type.
 * \return The constructed expression.
 */
TVM_DLL PrimExpr fast_erf_float_expr(PrimExpr arg, int bits);

inline void CheckMathUnaryOpInputDType(const char* op_name, const PrimType& dtype) {
  TVM_FFI_CHECK(dtype.code() == DLDataTypeCode::kDLFloat ||
                    dtype.MatchesElementType(DLDataTypeCode::kDLBfloat, 16),
                TypeError)
      << "tirx." << op_name << " only supports floating-point inputs, but got " << dtype;
}

// Intrinsic operators
#define TVM_DECLARE_INTRIN_UNARY_WITH_CHECK(OpName, CheckInputDType)                           \
  inline PrimExpr OpName(PrimExpr x, Span span = Span()) {                                     \
    static const Op op = Op::Get("tirx." #OpName);                                             \
    PrimType x_ty = x.ty();                                                                    \
    CheckInputDType(#OpName, x_ty);                                                            \
    if (x_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {                              \
      PrimType bf16_ty = x_ty;                                                                 \
      PrimType f32_ty =                                                                        \
          x_ty.IsScalableVector()                                                              \
              ? PrimType::ScalableVector(DLDataTypeCode::kDLFloat, 32, x_ty.VScaleFactor())    \
              : PrimType::Float(32, x_ty.lanes());                                             \
      PrimExpr x_fp32 = prim::Cast(f32_ty, x, span);                                           \
      PrimExpr result_fp32 = Call(f32_ty, op, {x_fp32}, {}, {}, span).as_or_throw<PrimExpr>(); \
      return prim::Cast(bf16_ty, result_fp32, span);                                           \
    } else {                                                                                   \
      return Call(x_ty, op, {x}, {}, {}, span).as_or_throw<PrimExpr>();                        \
    }                                                                                          \
  }

#define TVM_DECLARE_INTRIN_UNARY(OpName) \
  TVM_DECLARE_INTRIN_UNARY_WITH_CHECK(OpName, [](const char*, const PrimType&) {})

#define TVM_DECLARE_FLOAT_INTRIN_UNARY(OpName) \
  TVM_DECLARE_INTRIN_UNARY_WITH_CHECK(OpName, CheckMathUnaryOpInputDType)

TVM_DECLARE_INTRIN_UNARY(exp);
TVM_DECLARE_INTRIN_UNARY(exp2);
TVM_DECLARE_INTRIN_UNARY(exp10);
TVM_DECLARE_INTRIN_UNARY(erf);
TVM_DECLARE_FLOAT_INTRIN_UNARY(tanh);
TVM_DECLARE_INTRIN_UNARY(sigmoid);
TVM_DECLARE_INTRIN_UNARY(sqrt);
TVM_DECLARE_INTRIN_UNARY(rsqrt);
TVM_DECLARE_INTRIN_UNARY(log);
TVM_DECLARE_INTRIN_UNARY(log10);
TVM_DECLARE_INTRIN_UNARY(log1p);
TVM_DECLARE_INTRIN_UNARY(popcount);
TVM_DECLARE_FLOAT_INTRIN_UNARY(tan);
TVM_DECLARE_FLOAT_INTRIN_UNARY(cos);
TVM_DECLARE_FLOAT_INTRIN_UNARY(cosh);
TVM_DECLARE_FLOAT_INTRIN_UNARY(sin);
TVM_DECLARE_FLOAT_INTRIN_UNARY(sinh);
TVM_DECLARE_FLOAT_INTRIN_UNARY(asin);
TVM_DECLARE_FLOAT_INTRIN_UNARY(acos);
TVM_DECLARE_FLOAT_INTRIN_UNARY(atan);
TVM_DECLARE_FLOAT_INTRIN_UNARY(acosh);
TVM_DECLARE_FLOAT_INTRIN_UNARY(asinh);
TVM_DECLARE_FLOAT_INTRIN_UNARY(atanh);

#define TVM_DECLARE_INTRIN_BINARY(OpName)                                  \
  inline PrimExpr OpName(PrimExpr x, PrimExpr y, Span span = Span()) {     \
    static const Op op = Op::Get("tirx." #OpName);                         \
    return Call(x.ty(), op, {x, y}, {}, {}, span).as_or_throw<PrimExpr>(); \
  }

TVM_DECLARE_INTRIN_BINARY(atan2);
TVM_DECLARE_INTRIN_BINARY(nextafter);
TVM_DECLARE_INTRIN_BINARY(copysign);
TVM_DECLARE_INTRIN_BINARY(hypot);
TVM_DECLARE_INTRIN_BINARY(ldexp);

/*!
 * \brief Check if type is a pointer to a runtime element type.
 * \param type The type to be checked.
 * \param element_type The corresponding element type.
 * \return The check results
 */
inline bool IsPointerType(const Type& type, DLDataType element_type) {
  if (type.as<MissingType>().has_value()) return false;
  if (const auto* ptr_type = type.as<PointerTypeNode>()) {
    if (const auto* prim_type = ptr_type->element_type.as<PrimTypeNode>()) {
      return prim_type->dtype == element_type;
    }
  }
  return false;
}

/*!
 * \brief Make a constant opaque-pointer value.
 * \param value The integer payload to reinterpret as a handle.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
inline Expr ConstHandle(int64_t value, Span span = Span());

/*!
 * \brief Check whether stmt is nop.
 * \param stmt The input statement
 * \return whether stmt is nop
 */
inline bool is_no_op(const tirx::Stmt& stmt);

/*!
 * \brief Left fold.
 * \param freduce The reduction function.
 * \param init_value The initial value.
 * \param values The values to be folded.
 * \param span The location of the fold in the source.
 * \return The result.
 * \tparam FReduce The type of the reduction.
 */
template <typename FReduce>
inline PrimExpr foldl(FReduce freduce, PrimExpr init_value, const ffi::Array<PrimExpr>& values,
                      Span span = Span()) {
  for (PrimExpr val : values) {
    init_value = freduce(init_value, val, span);
  }
  return init_value;
}

inline bool is_no_op(const tirx::Stmt& stmt) {
  if (!stmt.defined()) return true;
  if (const auto* op = stmt.as<tirx::EvaluateNode>()) {
    auto value = op->value.as<PrimExpr>();
    return value && IsConstInt(value.value());
  }
  if (const auto* op = stmt.as<tirx::SeqStmtNode>()) {
    return op->seq.size() == 0;
  }
  return false;
}

inline Expr ConstHandle(int64_t value, Span span) {
  return reinterpret(PointerType::VoidPointerTy(), IntImm(PrimType::UInt(64), value, span), span);
}

TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(logaddexp);
}  // namespace tvm::prim

#endif  // TVM_TIR_OP_H_

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
 * \file tvm/tirx/op/abi.h
 * \brief Abi operations for TIRx.
 */
#ifndef TVM_TIRX_OP_ABI_H_
#define TVM_TIRX_OP_ABI_H_

#include <tvm/ir/attrs.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

namespace tvm::tirx {

/*!
 * \brief Call an external C function with argument and result types supplied by the caller.
 *
 * Arguments, in order:
 * - args[0]: func_name, The function name.
 * - args[1...]: args, trailing Expr operands.
 */
TVM_DLL const Op& call_extern_op();

/*!
 * \brief Call a pure external C function with types supplied by the caller.
 *
 * Arguments, in order:
 * - args[0]: func_name, The function name.
 * - args[1...]: args, trailing Expr operands.
 */
TVM_DLL const Op& call_pure_extern_op();

/*!
 * \brief Invoke an LLVM intrinsic.
 *
 * Arguments, in order:
 * - args[0]: intrin_id, The intrinsic identifier.
 * - args[1...]: args, trailing Expr operands.
 */
TVM_DLL const Op& call_llvm_intrin_op();

/*!
 * \brief Invoke a pure LLVM intrinsic.
 *
 * Arguments, in order:
 * - args[0]: intrin_id, The intrinsic identifier.
 * - args[1...]: args, trailing Expr operands.
 */
TVM_DLL const Op& call_llvm_pure_intrin_op();

/*!
 * \brief Invoke a pure GLSL450 SPIR-V intrinsic.
 *
 * Arguments, in order:
 * - args[0]: intrin_id, The intrinsic identifier.
 * - args[1...]: args, trailing PrimExpr operands.
 */
TVM_DLL const Op& call_spirv_pure_glsl450_op();

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
 * \brief Read a runtime structure field.
 *
 * Arguments, in order:
 * - args[0]: arr, The array.
 * - args[1]: index, The index.
 * - args[2]: field, The field index.
 */
TVM_DLL const Op& tvm_struct_get_op();

/*!
 * \brief Write a runtime structure field.
 *
 * Arguments, in order:
 * - args[0]: arr, The array.
 * - args[1]: index, The index.
 * - args[2]: field, The field index.
 * - args[3]: value, The value to use.
 */
TVM_DLL const Op& tvm_struct_set_op();

/*!
 * \brief Raise the last runtime error.
 */
TVM_DLL const Op& tvm_throw_last_error_op();

/*!
 * \brief Allocate stack storage for runtime values.
 *
 * Arguments, in order:
 * - args[0]: dtype_str, The data type name.
 * - args[1]: num, The number of entries.
 */
TVM_DLL const Op& tvm_stack_alloca_op();

/*!
 * \brief Construct a stack-allocated shape tuple.
 *
 * Arguments, in order:
 * - args[0...]: args, trailing IntExpr operands.
 */
TVM_DLL const Op& tvm_stack_make_shape_op();

/*!
 * \brief Construct a stack-allocated DLTensor.
 *
 * Arguments, in order:
 * - args[0]: data, The input data.
 * - args[1]: shape, The shape.
 * - args[2]: strides, The strides.
 * - args[3]: ndim, The number of dimensions.
 * - args[4]: arr_dtype, The array data type.
 * - args[5]: elem_offset, The element offset.
 */
TVM_DLL const Op& tvm_stack_make_array_op();

/*!
 * \brief Invoke a runtime packed function.
 *
 * Arguments, in order:
 * - args[0]: func_name, The function name.
 * - args[1...]: args, trailing Expr operands.
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

/*!
 * \brief Invoke a C packed function.
 *
 * Arguments, in order:
 * - args[0]: func_name, The function name.
 * - args[1...]: args, trailing Expr operands.
 */
TVM_DLL const Op& tvm_call_cpacked_op();

/*!
 * \brief Invoke a packed function using an explicit argument stack.
 *
 * Arguments, in order:
 * - args[0]: func_name, The function name.
 * - args[1]: args_stack, The argument stack.
 * - args[2]: begin, The start index.
 * - args[3]: end, The end index.
 */
TVM_DLL const Op& tvm_call_packed_lowered_op();

/*!
 * \brief Invoke a C packed function using an explicit argument stack.
 *
 * Arguments, in order:
 * - args[0]: func_name, The function name.
 * - args[1]: args_stack, The argument stack.
 * - args[2]: begin, The start index.
 * - args[3]: end, The end index.
 */
TVM_DLL const Op& tvm_call_cpacked_lowered_op();

}  // namespace tvm::tirx

#endif  // TVM_TIRX_OP_ABI_H_

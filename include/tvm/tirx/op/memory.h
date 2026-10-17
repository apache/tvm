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
 * \file tvm/tirx/op/memory.h
 * \brief Memory operations for TIRx.
 */
#ifndef TVM_TIRX_OP_MEMORY_H_
#define TVM_TIRX_OP_MEMORY_H_

#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

namespace tvm {
namespace tirx {

/*!
 * \brief Allocate a tensor with exactly one TensorType in ty_args.
 *
 * The optional args[0] is a Tuple of producer-owned placement coordinates.
 * DictAttrs directly holds allocation annotations. The TensorType owns all
 * shape, dtype, storage scope and access/layout metadata, and is the result type.
 */
TVM_TIRX_DLL const Op& alloc_tensor_op();

/*!
 * \brief Declare a tensor view with exactly one TensorType in ty_args.
 *
 * args[0] is the existing physical pointer backing the view. There are no
 * attributes. The TensorType is the result type; no memory is allocated.
 */
TVM_TIRX_DLL const Op& decl_tensor_op();

/*!
 * \brief Reinterpret the bits of a value as an explicit primitive or pointer type.
 *
 * Arguments, in order:
 * - args[0]: x, The input value.
 */
TVM_TIRX_DLL const Op& reinterpret_op();

/*!
 * \brief perform reinterpret cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note This function may return value if the type is the same.
 */
TVM_TIRX_DLL Expr reinterpret(Type target_ty, Expr value, Location loc = UnknownLoc());

inline PrimExpr reinterpret(PrimType t, PrimExpr value, Location loc = UnknownLoc()) {
  return reinterpret(Type(std::move(t)), Expr(std::move(value)), std::move(loc))
      .as_or_throw<PrimExpr>();
}

/*! \brief Construct an opaque pointer from its integer payload. */
inline Expr ConstHandle(int64_t value, Location loc = UnknownLoc()) {
  return reinterpret(PtrType::VoidPtrType(), IntImm(PrimType::UInt(64), value, loc), loc);
}

/*!
 * \brief Return the address of a tensor element or addressable variable.
 *
 * Arguments, in order:
 * - args[0]: obj, The referenced object.
 */
TVM_TIRX_DLL const Op& address_of_op();

/*!
 * \brief Test whether a pointer is null.
 *
 * Arguments, in order:
 * - args[0]: x, The input value.
 */
TVM_TIRX_DLL const Op& isnullptr_op();

/*!
 * \brief Prefetch a memory address.
 *
 * Arguments, in order:
 * - args[0]: ptr, The pointer.
 * - args[1]: rw, The read/write mode.
 * - args[2]: locality, The locality hint.
 * - args[3]: cache_type, The cache policy.
 */
TVM_TIRX_DLL const Op& prefetch_op();

/*!
 * \brief Cast a handle to a typed pointer after adding a byte offset.
 *
 * Arguments, in order:
 * - args[0]: data, Base pointer.
 * - args[1]: byte_offset, Offset in bytes.
 * The result type specifies the pointed-to element type.
 */
TVM_TIRX_DLL const Op& ptr_byte_offset_op();

/*!
 * \brief Create a function local static handle that iniitalizes to nullptr.
 */
TVM_TIRX_DLL const Op& static_handle_op();

/*!
 * \brief Add a byte offset to an opaque pointer.
 *
 * Arguments, in order:
 * - args[0]: handle, The handle.
 * - args[1]: offset, The offset.
 */
TVM_TIRX_DLL const Op& handle_add_byte_offset_op();

/*!
 * \brief atomic add instruction, corresponding e.g. to atomicAdd in CUDA.
 *
 * Arguments, in order:
 * - args[0]: ptr, The pointer.
 * - args[1]: value, The value to use.
 */
TVM_TIRX_DLL const Op& atomic_add_op();

/*!
 * \brief Assume a tensor's base address has the given constant byte alignment.
 *
 * This leaf operation carries a compiler fact, without checking or changing
 * the address. Alignment must be a power of two between 1 and 2^27 bytes.
 */
TVM_TIRX_DLL const Op& assume_aligned_op();

/*!
 * \brief Returns an initialized but arbitrary value.
 */
TVM_TIRX_DLL const Op& undef_op();

/*!
 * \brief Masked tensor load.
 *
 * Arguments, in order:
 * - args[0]: tensor, The tensor.
 * - args[1...n-2]: The tensor indices (at least one).
 * - args[n-1]: The boolean lane mask, matching the lanes of the loaded vector.
 */
TVM_TIRX_DLL const Op& masked_load_op();

/*!
 * \brief Masked tensor store.
 *
 * Arguments, in order:
 * - args[0]: tensor, The tensor.
 * - args[1]: value, The vector value to store.
 * - args[2...n-2]: The tensor indices (at least one).
 * - args[n-1]: The boolean lane mask, matching the lanes of value.
 */
TVM_TIRX_DLL const Op& masked_store_op();

/*!
 * \brief Project the physical pointer associated with a TensorVar definition.
 *
 * Argument: tensor, the TensorVar whose physical pointer is projected.
 * The result pointer type is derived from the TensorType dtype and storage
 * scope of the sole TensorVar argument.  This operation is consumed by TIRx
 * lowering and code generation.
 */
TVM_TIRX_DLL const Op& tensor_data_ptr_op();

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_OP_MEMORY_H_

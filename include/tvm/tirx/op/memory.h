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
 */
TVM_DLL const Op& decl_tensor_op();

/*!
 * \brief Reinterpret the bits of a value as an explicit primitive or pointer type.
 *
 * Arguments, in order:
 * - args[0]: x, The input value.
 */
TVM_DLL const Op& reinterpret_op();

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
TVM_DLL Expr reinterpret(Type target_ty, Expr value, Span span = Span());

/*! \brief Construct an opaque pointer from its integer payload. */
inline Expr ConstHandle(int64_t value, Span span = Span()) {
  return reinterpret(PointerType::VoidPointerTy(), IntImm(PrimType::UInt(64), value, span), span);
}

/*!
 * \brief Return the address of a tensor element or addressable variable.
 *
 * Arguments, in order:
 * - args[0]: obj, The referenced object.
 */
TVM_DLL const Op& address_of_op();

/*!
 * \brief Test whether a pointer is null.
 *
 * Arguments, in order:
 * - args[0]: x, The input value.
 */
TVM_DLL const Op& isnullptr_op();

/*!
 * \brief Prefetch a memory address.
 *
 * Arguments, in order:
 * - args[0]: ptr, The pointer.
 * - args[1]: rw, The read/write mode.
 * - args[2]: locality, The locality hint.
 * - args[3]: cache_type, The cache policy.
 */
TVM_DLL const Op& prefetch_op();

/*!
 * \brief Get head access address with memory access pattern info.
 *
 * Arguments, in order:
 * - args[0]: data, The input data.
 * - args[1]: offset, The offset in access elements.
 * - args[2]: extent, The extent in access elements.
 * - args[3]: rw_mask, The read/write mask.
 * - ty_args[0]: The independent access element type.
 */
TVM_DLL const Op& access_ptr_op();

/*!
 * \brief Cast a handle to a typed pointer after adding a byte offset.
 *
 * Arguments, in order:
 * - args[0]: data, Base pointer.
 * - args[1]: byte_offset, Offset in bytes.
 * The result type specifies the pointed-to element type.
 */
TVM_DLL const Op& ptr_byte_offset_op();

/*!
 * \brief Create a function local static handle that iniitalizes to nullptr.
 */
TVM_DLL const Op& static_handle_op();

/*!
 * \brief Add a byte offset to an opaque pointer.
 *
 * Arguments, in order:
 * - args[0]: handle, The handle.
 * - args[1]: offset, The offset.
 */
TVM_DLL const Op& handle_add_byte_offset_op();

/*!
 * \brief atomic add instruction, corresponding e.g. to atomicAdd in CUDA.
 *
 * Arguments, in order:
 * - args[0]: ptr, The pointer.
 * - args[1]: value, The value to use.
 */
TVM_DLL const Op& atomic_add_op();

/*!
 * \brief Create an Nd memory allocation with storage scope.
 *
 * Arguments, in order:
 * - args[0]: storage_scope, The storage scope.
 * - args[1]: ndim, The number of dimensions.
 * - args[2]: shape, The shape.
 * - args[3...]: args, trailing Expr operands.
 */
TVM_DLL const Op& nd_mem_alloc_with_scope_op();

/*!
 * \brief Assume a tensor's base address has the given constant byte alignment.
 *
 * This leaf operation carries a compiler fact, without checking or changing
 * the address. Alignment must be a power of two between 1 and 2^27 bytes.
 */
TVM_DLL const Op& assume_aligned_op();

/*!
 * \brief Returns an initialized but arbitrary value.
 */
TVM_DLL const Op& undef_op();

/*!
 * \brief Masked buffer load.
 *
 * Arguments, in order:
 * - args[0]: buffer, The buffer.
 * - args[1]: index, The index.
 * - args[2...]: args, trailing PrimExpr operands.
 */
TVM_DLL const Op& masked_load_op();

/*!
 * \brief Masked buffer store.
 *
 * Arguments, in order:
 * - args[0]: buffer, The buffer.
 * - args[1]: value, The value to use.
 * - args[2]: index, The index.
 * - args[3...]: args, trailing PrimExpr operands.
 */
TVM_DLL const Op& masked_store_op();

/*!
 * \brief Get the element offset of a buffer given logical indices.
 *
 * Arguments, in order:
 * - args[0]: load, The buffer load whose offset is returned.
 */
TVM_DLL const Op& buffer_offset_op();

/*!
 * \brief Project the physical pointer associated with a TensorVar definition.
 *
 * Argument: tensor, the TensorVar whose physical pointer is projected.
 * The result pointer type is derived from the TensorType dtype and storage
 * scope of the sole TensorVar argument.  This operation is consumed by TIRx
 * lowering and code generation.
 */
TVM_DLL const Op& tensor_data_ptr_op();

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_OP_MEMORY_H_

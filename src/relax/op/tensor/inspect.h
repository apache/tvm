/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  Sex The NOTICE file
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
 * KIND, either express or implied.  Sex The License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file inspect.h
 * \brief Operators to access runtime DLTensor parameters
 */
#ifndef TVM_RELAX_OP_TENSOR_INSPECT_H_
#define TVM_RELAX_OP_TENSOR_INSPECT_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {
namespace inspect {

/* \brief Return the DLTensor::dtype::type_code field
 *
 * \param expr The relax expression to be inspected.  Must have
 * `TensorStructInfo`.
 *
 * \returns The uint8_t value of the type_code, with
 * `PrimStructInfo(DataType::UInt(8))`
 */
Expr tensor_dtype_code(Expr expr);

/* \brief Return the DLTensor::dtype::bits field
 *
 * \param expr The relax expression to be inspected.  Must have
 * `TensorStructInfo`.
 *
 * \returns The uint8_t value of the number of bits, with
 * `PrimStructInfo(DataType::UInt(8))`.  For vectorized types, returns
 * the bit width of the underlying scalar type (e.g. 32 for
 * "float32x4", not 128).
 */
Expr tensor_dtype_bits(Expr expr);

/* \brief Return the DLTensor::dtype::lanes field
 *
 * \param expr The relax expression to be inspected.  Must have
 * `TensorStructInfo`.
 *
 * \returns The uint16_t value of the number of lanes, with
 * `PrimStructInfo(DataType::UInt(16))`
 */
Expr tensor_dtype_lanes(Expr expr);

/* \brief Return the DLTensor::ndim field
 *
 * \param expr The relax expression to be inspected.  Must have
 * `TensorStructInfo`.
 *
 * \returns The int32_t value of the dimensionality, with
 * `PrimStructInfo(DataType::Int(32))`.
 */
Expr tensor_ndim(Expr expr);

/* \brief Return the DLTensor::shape[i] field
 *
 * \param expr The relax expression to be inspected.  Must have
 * `TensorStructInfo`.
 *
 * \param axis The axis to inspect.  Must be within the range `0 <=
 *     axis < tensor_ndim(expr)`, or else the results are undefined.
 *
 * \returns The int64_t extent of the specified tensor axis, with
 * `PrimStructInfo(DataType::Int(64))`.
 */
Expr tensor_shape_i(Expr expr, Expr axis);

/* \brief Return the DLTensor::strides[i] field
 *
 * The `int64_t* DLTensor::strides` is allowed to be NULL, which
 * represents a compact packing of the data.  In this case, the
 * returned stride is computed from the `DLTensor::shape`.
 *
 * \param expr The relax expression to be inspected.  Must have
 * `TensorStructInfo`.
 *
 * \param axis The axis to inspect.  Must be within the range `0 <=
 *     axis < tensor_ndim(expr)`, or else the results are undefined.
 *
 * \returns The int64_t extent of the specified tensor axis, with
 * `PrimStructInfo(DataType::Int(64))`.
 */
Expr tensor_stride_i(Expr expr, Expr axis);

/* \brief Return the DLTensor::byte_offset field
 *
 * \param expr The relax expression to be inspected.  Must have
 * `TensorStructInfo`.
 *
 * \returns The uint64_t byte offset, with `PrimStructInfo(DataType::UInt(64))`.
 */
Expr tensor_byte_offset(Expr expr);

/* \brief Return the element offset of a DLTensor
 *
 * While the DLTensor does not directly contain the element offset, it
 * can be inferred from the `DLTensor::byte_offset` and
 * `DLTensor::data_type` fields.
 *
 * \param expr The relax expression to be inspected.  Must have
 * `TensorStructInfo`.
 *
 * \returns The uint64_t element offset, with `PrimStructInfo(DataType::UInt(64))`.
 */
Expr tensor_elem_offset(Expr expr);

}  // namespace inspect
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_INSPECT_H_

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
 * \file manipulate.h
 * \brief The functions to make Relax tensor manipulation operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_MANIPULATE_H_
#define TVM_RELAX_OP_TENSOR_MANIPULATE_H_

#include <tvm/relax/attrs/manipulate.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*! \brief Broadcasts a tensor to a specified shape. */
Expr broadcast_to(Expr x, Expr shape);

/*!
 * \brief Concatenate the input tensors along the given axis.
 * \param tensors An Expr in Tuple type, containing the tensors to be concatenated,
 * or a list of tensors
 * \param axis The axis along which the tensors are concatenated.
 * If it is `NullOpt`, the input tensor is required to be flattened before concatenation.
 * \return The concatenated tensor.
 */
Expr concat(Expr tensors, Optional<Integer> axis);

/*!
 * \brief Insert new axes at the positions given by `axis`.
 * \param x The input data to the operator.
 * \param axis The axes at which the input array are expanded.
 * \return The transformed result.
 */
Expr expand_dims(Expr x, Array<Integer> axis);

/*!
 * \brief Flatten all the tensor dimensions into one.
 * \param x The input data to the operator.
 * \return The flattened result.
 */
Expr flatten(Expr x);

/*!
 * \brief Transform layout of a tensor.
 * \param x The input data to the operator.
 * \param index_map The transformation to apply.
 * \param pad_value The value used for padding if the transformation results in implicit padding. If
 * not specified, any value can be used.
 * \return The transformed result.
 */
Expr layout_transform(Expr x, tir::IndexMap index_map, Optional<PrimValue> pad_value);

/*!
 * \brief Permutes the dimensions of an array.
 * \param x The input data to the operator.
 * \param axes The target axes order, reverse order if not specified.
 * \return The transposed result.
 */
Expr permute_dims(Expr x, Optional<Array<Integer>> axes);

/*!
 * \brief Reshape the input array, supporting `-1` inference in the new
 * shape when the new shape is given as an Array of PrimExpr.
 * \param x The input data to the operator.
 * \param shape The new shape. Should be compatible with the original shape.
 * It is required to be either an Array of PrimExpr, or a Shape in Relax
 * \return The reshaped result.
 */
Expr reshape(Expr x, ObjectRef shape);

/*!
 * \brief Split input tensor along axis by sections or indices.
 * - If indices_or_sections is an integer, the input will be divided equally
 * along given axis (if possible). Last section will be smaller if the tensor
 * size along the given dimension is not divisible by the integer.
 * - If indices_or_sections is a tuple of mixture of int or PrimExpr,
 * the entries indicate the indices where along axis the array is split.
 * \param x The tensor to be split.
 * \param indices_or_sections Indices or sections to split into.
 * It is required to be an Array of PrimExpr or an integer.
 * \param axis The axis over which to split.
 * \return The computed result.
 */
Expr split(Expr x, ObjectRef indices_or_sections, int axis);

/*!
 * \brief Squeeze axes in the array.
 * \param x The input data to the operator.
 * \param axis The set of axes to remove.
 * If it is `NullOpt`, remove all axis of dimensions 1.
 * If any specified axis has dimension that does not equal 1, it is an error.
 * \return The squeezed result.
 */
Expr squeeze(Expr x, Optional<Array<Integer>> axis);

/*!
 * \brief Return a summation of data to the shape of collapse_target.
 * For details, please see the operator `relax.collapse_sum_to`.
 * \param data The input tensor.
 * \param collapse_target The tensor whose shape is the shape to collapse to.
 * \return The result tensor after summation.
 */
Expr collapse_sum_like(Expr data, Expr collapse_target);

/*!
 * \brief Return a summation of data to the given shape.
 * collapse_sum_to is intended as the backward operator of broadcast_to and
 * other broadcast operators in the automatic differentiation process.
 * We expect that data is the result of broadcasting some tensor of the given shape in some
 * broadcast operation. Thus the given shape and data.shape must follow broadcast rules.
 * \param data The input tensor.
 * \param shape The shape to collapse to.
 * \return The result tensor of the given shape after summation.
 */
Expr collapse_sum_to(Expr data, Expr shape);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_MANIPULATE_H_

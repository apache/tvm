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
 * \file tvm/tirx/op/tile.h
 * \brief TIRX tile primitive statements, operators, and reified lambda expressions.
 */
#ifndef TVM_TIRX_OP_TILE_H_
#define TVM_TIRX_OP_TILE_H_

#include <tvm/ffi/object.h>
#include <tvm/ir/op.h>

namespace tvm {
namespace tirx {

/*!
 * \brief The type of the function that sanitizes the arguments of a TIRX operator.
 * \param op The operator.
 * \param args The arguments.
 */
using FArgSanitizer = ffi::TypedFunction<void(tvm::Op, ffi::Array<ffi::ObjectRef>)>;

/*! \brief Tile primitive operator handles. */
namespace tile {

/*!
 * \brief See pesudo code below:
 *
 * Tx.cast(TensorRegion dst, TensorRegion src)
 */
TVM_DLL const Op& cast_op();

/*!
 * \brief See pesudo code below:
 *
 * Tx.copy(TensorRegion dst, TensorRegion src)
 */
TVM_DLL const Op& copy_op();

/*!
 * \brief See pesudo code below:
 *
 * Tx.Async.copy(TensorRegion dst, TensorRegion src)
 */
TVM_DLL const Op& copy_async_op();

/*!
 * \brief See pesudo code below:
 *
 *  Tx.fill(TensorRegion dst, PrimExpr value)
 */
TVM_DLL const Op& fill_op();

/*!
 * \brief See pesudo code below:
 *
 * Tx.gemm(TensorVar A, TensorVar B, TensorVar C, TensorVar D, PrimExpr alpha, PrimExpr beta)
 */
TVM_DLL const Op& gemm_op();

/*!
 * \brief See pesudo code below:
 *
 * Tx.gemm_async(TensorRegion C, TensorRegion A, TensorRegion B, bool transA, bool transB,
 * bool accum)
 */
TVM_DLL const Op& gemm_async_op();

/*!
 * \brief Get the tile zero operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 */
TVM_DLL const Op& zero_op();

/*!
 * \brief Get the tile sqrt operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 */
TVM_DLL const Op& sqrt_op();
/*!
 * \brief Get the tile sqrt with scale bias operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 * - args[2]: scale, The scale factor.
 * - args[3]: bias, The bias.
 */
TVM_DLL const Op& sqrt_with_scale_bias_op();

/*!
 * \brief Get the tile exp operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 */
TVM_DLL const Op& exp_op();
/*!
 * \brief Get the tile exp with scale bias operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 * - args[2]: scale, The scale factor.
 * - args[3]: bias, The bias.
 */
TVM_DLL const Op& exp_with_scale_bias_op();

/*!
 * \brief Get the tile exp2 operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 */
TVM_DLL const Op& exp2_op();
/*!
 * \brief Get the tile exp2 with scale bias operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 * - args[2]: scale, The scale factor.
 * - args[3]: bias, The bias.
 */
TVM_DLL const Op& exp2_with_scale_bias_op();

/*!
 * \brief Get the tile log2 operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 */
TVM_DLL const Op& log2_op();
/*!
 * \brief Get the tile log2 with scale bias operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 * - args[2]: scale, The scale factor.
 * - args[3]: bias, The bias.
 */
TVM_DLL const Op& log2_with_scale_bias_op();

/*!
 * \brief Get the tile add operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src1, The first source.
 * - args[2]: src2, The second source.
 */
TVM_DLL const Op& add_op();

/*!
 * \brief Get the tile sub operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src1, The first source.
 * - args[2]: src2, The second source.
 */
TVM_DLL const Op& sub_op();

/*!
 * \brief Get the tile mul operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src1, The first source.
 * - args[2]: src2, The second source.
 */
TVM_DLL const Op& mul_op();

/*!
 * \brief Get the tile fdiv operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src1, The first source.
 * - args[2]: src2, The second source.
 */
TVM_DLL const Op& fdiv_op();

/*!
 * \brief Get the tile minimum operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src1, The first source.
 * - args[2]: src2, The second source.
 */
TVM_DLL const Op& minimum_op();

/*!
 * \brief Get the tile maximum operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src1, The first source.
 * - args[2]: src2, The second source.
 */
TVM_DLL const Op& maximum_op();

/*!
 * \brief Get the tile reciprocal operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 */
TVM_DLL const Op& reciprocal_op();

/*!
 * \brief Get the tile sum operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 * - args[2]: axes, The axes.
 * - args[3]: accum, The accumulator.
 */
TVM_DLL const Op& sum_op();

/*!
 * \brief Get the tile max operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 * - args[2]: axes, The axes.
 * - args[3]: accum, The accumulator.
 */
TVM_DLL const Op& max_op();

/*!
 * \brief Get the tile min operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 * - args[2]: axes, The axes.
 * - args[3]: accum, The accumulator.
 */
TVM_DLL const Op& min_op();

/*!
 * \brief Get the tile memset operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: value, The value to use.
 */
TVM_DLL const Op& memset_op();

/*!
 * \brief Get the tile reduce negate operation.
 *
 * Arguments, in order:
 * - args[0]: output, The output.
 * - args[1]: input, The input.
 * - args[2]: reduce_op, The reduction operation.
 * - args[3]: reduce_axes, The reduction axes.
 * - args[4]: accum, The accumulator.
 */
TVM_DLL const Op& reduce_negate_op();

/*!
 * \brief Get the tile binary reduce operation.
 *
 * Arguments, in order:
 * - args[0]: binary_output, The binary operation output.
 * - args[1]: reduce_output, The reduction output.
 * - args[2]: binary_input1, The first binary input.
 * - args[3]: binary_input2, The second binary input.
 * - args[4]: binary_op, The binary operation.
 * - args[5]: reduce_op, The reduction operation.
 * - args[6]: reduce_axes, The reduction axes.
 */
TVM_DLL const Op& binary_reduce_op();

/*!
 * \brief Get the tile unary reduce operation.
 *
 * Arguments, in order:
 * - args[0]: unary_output, The unary operation output.
 * - args[1]: reduce_output, The reduction output.
 * - args[2]: unary_input, The unary operation input.
 * - args[3]: unary_op, The unary operation.
 * - args[4]: reduce_op, The reduction operation.
 * - args[5]: reduce_axes, The reduction axes.
 */
TVM_DLL const Op& unary_reduce_op();
/*!
 * \brief Get the tile unary reduce with scale bias operation.
 *
 * Arguments, in order:
 * - args[0]: unary_output, The unary operation output.
 * - args[1]: reduce_output, The reduction output.
 * - args[2]: unary_input, The unary operation input.
 * - args[3]: unary_op, The unary operation.
 * - args[4]: reduce_op, The reduction operation.
 * - args[5]: scale, The scale factor.
 * - args[6]: bias, The bias.
 * - args[7]: reduce_axes, The reduction axes.
 */
TVM_DLL const Op& unary_reduce_with_scale_bias_op();

/*!
 * \brief Get the tile binary chain operation.
 *
 * Arguments, in order:
 * - args[0]: output, The output.
 * - args[1]: data, The input data.
 * - args[2]: operand0, The first operand.
 * - args[3]: operand1, The second operand.
 * - args[4]: op0, The first operation.
 * - args[5]: op1, The second operation.
 * - args[6]: reverse1, Whether to reverse the second operation.
 */
TVM_DLL const Op& binary_chain_op();

/*!
 * \brief Get the tile select operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: true_value, The value when the condition is true.
 * - args[2]: false_value, The value when the condition is false.
 * - args[3]: pred, The predicate.
 */
TVM_DLL const Op& select_op();

/*!
 * \brief Get the tile fma operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 * - args[2]: scale, The scale factor.
 * - args[3]: bias, The bias.
 */
TVM_DLL const Op& fma_op();

/*!
 * \brief Get the tile silu operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 */
TVM_DLL const Op& silu_op();

/*!
 * \brief Get the tile permute layout operation.
 *
 * Arguments, in order:
 * - args[0]: dst, The destination.
 * - args[1]: src, The source.
 */
TVM_DLL const Op& permute_layout_op();

}  // namespace tile

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_OP_TILE_H_

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

TVM_DLL const Op& zero_op();

TVM_DLL const Op& sqrt_op();
TVM_DLL const Op& sqrt_with_scale_bias_op();

TVM_DLL const Op& exp_op();
TVM_DLL const Op& exp_with_scale_bias_op();

TVM_DLL const Op& exp2_op();
TVM_DLL const Op& exp2_with_scale_bias_op();

TVM_DLL const Op& log2_op();
TVM_DLL const Op& log2_with_scale_bias_op();

TVM_DLL const Op& add_op();

TVM_DLL const Op& sub_op();

TVM_DLL const Op& mul_op();

TVM_DLL const Op& fdiv_op();

TVM_DLL const Op& minimum_op();

TVM_DLL const Op& maximum_op();

TVM_DLL const Op& reciprocal_op();

TVM_DLL const Op& sum_op();

TVM_DLL const Op& max_op();

TVM_DLL const Op& min_op();

TVM_DLL const Op& memset_op();

TVM_DLL const Op& reduce_negate_op();

TVM_DLL const Op& binary_reduce_op();

TVM_DLL const Op& unary_reduce_op();
TVM_DLL const Op& unary_reduce_with_scale_bias_op();

TVM_DLL const Op& binary_chain_op();

TVM_DLL const Op& select_op();

TVM_DLL const Op& fma_op();

TVM_DLL const Op& silu_op();

TVM_DLL const Op& permute_layout_op();

}  // namespace tile

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_OP_TILE_H_

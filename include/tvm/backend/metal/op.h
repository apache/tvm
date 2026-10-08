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
 * \file tvm/backend/metal/op.h
 * \brief Metal-owned matrix operations.
 */
#ifndef TVM_BACKEND_METAL_OP_H_
#define TVM_BACKEND_METAL_OP_H_

#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

namespace tvm::backend::metal {

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

}  // namespace tvm::backend::metal

#endif  // TVM_BACKEND_METAL_OP_H_

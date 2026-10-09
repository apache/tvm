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
 * \file tvm/backend/metal/op/cooperative_tensor.h
 * \brief Metal-owned matrix operations.
 */
#ifndef TVM_BACKEND_METAL_OP_COOPERATIVE_TENSOR_H_
#define TVM_BACKEND_METAL_OP_COOPERATIVE_TENSOR_H_

#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

namespace tvm {
namespace backend {
namespace metal {

/*!
 * \brief Fill a cooperative_tensor with a given value.
 *
 * Arguments, in order:
 * - args[0]: d, The D operand.
 * - args[1]: index, The index.
 * - args[2]: value, The value to use.
 * - args[3]: rows, The number of rows.
 * - args[4]: cols, The number of columns.
 */
TVM_DLL const Op& cooperative_tensor_fill_op();

/*!
 * \brief Load data from device or threadgroup memory into a cooperative_tensor.
 *
 * Arguments, in order:
 * - args[0]: d, The D operand.
 * - args[1]: index, The index.
 * - args[2]: ptr, The pointer.
 * - args[3]: stride, The stride.
 * - args[4]: rows, The number of rows.
 * - args[5]: cols, The number of columns.
 * - args[6]: transpose_matrix, Whether to transpose the matrix.
 * - args[7]: mma_M, The M dimension of the matrix operation.
 * - args[8]: mma_N, The N dimension of the matrix operation.
 * - args[9]: mma_K, The K dimension of the matrix operation.
 * - args[10]: operand_role, 0 for left (A), 1 for right (B), 2 for destination (C).
 */
TVM_DLL const Op& cooperative_tensor_load_op();

/*!
 * \brief Store data from a cooperative_tensor to device or threadgroup memory.
 *
 * Arguments, in order:
 * - args[0]: d, The D operand.
 * - args[1]: index, The index.
 * - args[2]: ptr, The pointer.
 * - args[3]: stride, The stride.
 * - args[4]: rows, The number of rows.
 * - args[5]: cols, The number of columns.
 * - args[6]: transpose_matrix, Whether to transpose the matrix.
 * - args[7]: mma_M, The M dimension of the matrix operation.
 * - args[8]: mma_N, The N dimension of the matrix operation.
 * - args[9]: mma_K, The K dimension of the matrix operation.
 * - args[10]: operand_role, 0 for left (A), 1 for right (B), 2 for destination (C).
 */
TVM_DLL const Op& cooperative_tensor_store_op();

/*!
 * \brief Multiply and accumulate two matrices using cooperative_tensor
 *        (MetalPerformancePrimitives matmul2d).
 *
 * Arguments, in order:
 * - args[0]: d, The D operand.
 * - args[1]: index_d, The D fragment index.
 * - args[2]: a, The A operand.
 * - args[3]: index_a, The A fragment index.
 * - args[4]: b, The B operand.
 * - args[5]: index_b, The B fragment index.
 * - args[6]: c, The C operand.
 * - args[7]: index_c, The C fragment index.
 * - args[8]: M, The M dimension.
 * - args[9]: N, The N dimension.
 * - args[10]: K, The K dimension.
 * - args[11]: transpose_a, Whether to transpose A.
 * - args[12]: transpose_b, Whether to transpose B.
 */
TVM_DLL const Op& cooperative_tensor_multiply_accumulate_op();

}  // namespace metal
}  // namespace backend
}  // namespace tvm

#endif  // TVM_BACKEND_METAL_OP_COOPERATIVE_TENSOR_H_

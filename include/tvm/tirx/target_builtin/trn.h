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
 * \file tvm/tir/target_builtin/trn.h
 * \brief TIR builtin intrinsics specific to Trainium target.
 */
#ifndef TVM_TIRX_TARGET_BUILTIN_TRN_H_
#define TVM_TIRX_TARGET_BUILTIN_TRN_H_

#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tirx {
namespace builtin {

/*!
 * \brief nki intrinsics for load operation.
 *
 * nki_load(result, data)
 */
TVM_DLL const Op& nki_load();
/*!
 * \brief nki intrinsics for store operation.
 *
 * nki_store(result, data)
 */
TVM_DLL const Op& nki_store();
/*!
 * \brief nki intrinsics for tensor_copy operation.
 *
 * nki_tensor_copy(result, data)
 */
TVM_DLL const Op& nki_tensor_copy();
/*!
 * \brief nki intrinsics for matmul operation.
 *
 * nki_matmul(C, A, B, accum)
 *
 * equivalent to C += A.T @ B (if accum is true), or C = A.T @ B (if accum is false)
 */
TVM_DLL const Op& nki_matmul();

/*!
 * \brief nki intrinsics for activation operation.
 *
 * nki_activation(result, data, opcode, bias, scale)
 */
TVM_DLL const Op& nki_activation();

/*!
 * \brief nki intrinsics for reciprocal operation.
 *
 * nki_reciprocal(result, data)
 */
TVM_DLL const Op& nki_reciprocal();

/*!
 * \brief nki intrinsics for tensortensor operation.
 *
 * nki_tensortensor(result, operand0, operand1, opcode)
 */
TVM_DLL const Op& nki_tensortensor();

/*!
 * \brief nki intrinsics for tensorscalar operation.
 *
 * nki_tensorscalar(result, operand0, operand1, opcode, reverse)
 */
TVM_DLL const Op& nki_tensorscalar();

/*!
 * \brief nki intrinsics for tensorreduce operation.
 *
 * nki_tensorreduce(result, data, opcode, negate, axes)
 */
TVM_DLL const Op& nki_tensorreduce();

/*!
 * \brief nki intrinsics for memset operation.
 *
 * nki_memset(result, value)
 */
TVM_DLL const Op& nki_memset();

/*!
 * \brief nki intrinsics for activation reduce operation.
 *
 * nki_activation_reduce(reduce_res, act_res, data, opcode, reduce_opcode, bias, scale)
 */
TVM_DLL const Op& nki_activation_reduce();

/*!
 * \brief nki intrinsics for tensorscalar reduce operation.
 *
 * nki_tensorscalar_reduce(reduce_res, tensorscalar_res, operand0, operand1, opcode, reduce_opcode,
 * reverse)
 */
TVM_DLL const Op& nki_tensorscalar_reduce();

/*!
 * \brief nki intrinsics for initializing identity tensor.
 *
 * nki_identity(result, size)
 */
TVM_DLL const Op& nki_identity();

/*!
 * \brief nki intrinsics for scalar tensor tensor operation.
 *
 * (data op1 operand1) op2 (operand2) where op1 is tensor-scalar and op2 is tensor-tensor
 *
 * nki_scalar_tensor_tensor(result, data, operand0, operand1, opcode0, opcode1, reverse0, reverse1)
 *
 */
TVM_DLL const Op& nki_scalar_tensor_tensor();

/*!
 * \brief nki intrinsics for scalar tensor scalar operation.
 *
 * (data op1 operand1) op2 (operand2) where op1 and op2 are tensor-scalar
 *
 * nki_scalar_tensor_scalar(result, data, operand0, operand1, opcode0, opcode1, reverse0, reverse1)
 *
 */
TVM_DLL const Op& nki_scalar_tensor_scalar();

/*!
 * \brief nki intrinsics for affine_select operation.
 *
 * nki_affine_select(result, pred, true_value, false_value)
 */
TVM_DLL const Op& nki_affine_select();

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_TARGET_BUILTIN_TRN_H_

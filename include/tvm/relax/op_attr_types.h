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
 * \file tvm/relax/op_attr_types.h
 * \brief Data structures that can appear in operator attributes.
 */
#ifndef TVM_RELAX_OP_ATTR_TYPES_H_
#define TVM_RELAX_OP_ATTR_TYPES_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/te/tensor.h>

namespace tvm {
namespace relax {

enum OpPatternKind {
  // Elementwise operation
  kElemWise = 0,
  // Broadcasting operator, can always map output axis to the input in order.
  // for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
  // Note that the axis need to be in order so transpose is not a bcast operator.
  kBroadcast = 1,
  // Injective operator, can always injectively map output axis to a single input axis.
  // All injective operator can still be safely fused to injective and reduction.
  kInjective = 2,
  // Communicative reduction operator.
  kCommReduce = 3,
  // Complex operation, can still fuse elemwise operations into its output.
  // but cannot chain another complex op
  kOutEWiseFusable = 4,
  // The pattern for tuple nodes. Can fuse into subsequent injective ops,
  // but treated specially
  kTuple = 7,
  // Opaque operation, cannot fuse anything.
  kOpaque = 8
};

/*!
 * \brief Infer output struct info given the call
 *
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 */
using FInferStructInfo = ffi::TypedFunction<StructInfo(const Call& call, const BlockBuilder& ctx)>;

/*!
 * \brief Packed function implementation for operators. The relax operator will be lowered to
 * this packed function call during codegen.
 */
using FCallPacked = String;

/*!
 * \brief The function type of a normalization function.
 *
 * A normalization function is used when a `relax::Call` may be
 * expressed in multiple syntactically valid and semantically
 * equivalent forms, to normalize to a single representation.
 *
 * Note: `FNormalize` is applied for each expression as part of the
 *    `relax::BlockBuilder`.  While operator-specific validation may
 *    be performed within the `FNormalize` implementation, ensuring
 *    that errors are caught as early as possible, this should only be
 *    used when validation is fast to apply.  If the validation logic
 *    may be slow, it should instead be implemented in `FValidate`,
 *    which is only run as part of the well-formed checker.
 *
 * \param bb The BlockBuilder context.
 *
 * \param call The call to be normalized.  It is provided by-value, to
 * avoid copies for the common case where the call is already normalized.
 */
using FNormalize = ffi::TypedFunction<Expr(const BlockBuilder& bb, Call call)>;

/*!
 * \brief The function type of a validation function.
 *
 * A validation function is used to define constraints that should be
 * verified for an operator as part of the well-formed checker.
 *
 * Note: `FValidate` is only applied as part of the well-formed
 *    checker.  While this minimizes overhead while compiling Relax,
 *    this delay between generating an ill-formed `relax::Call` and
 *    identifying the ill-formed call may complicate debugging.  If
 *    the validation logic is very fast to check, and doing so would
 *    not introduce a significant overhead, consider validating as part
 *    of `FNormalize`, which is applied by the block builder for each
 *    `relax::Call`.
 *
 * \param call The call to be validated.
 */
using FValidate = ffi::TypedFunction<void(const Call& call)>;

/*! \brief The function type of a legalization function.
 *
 * A legalization function is used to replace a `relax::Call` with
 * more concrete implementations.  For example, the operation
 * `relax.op.add` may be replaced with a call to a TIR function
 * implementing addition of two tensors.
 *
 * The purpose of `FLegalize` is to remove calls to the operator while
 * lowering.  Therefore, unlike `FNormalize`, the resulting expression
 * may *not* contain the original operator.
 *
 * \param bb The BlockBuilder context.
 * \param call The call to be legalized.
 */
using FLegalize = ffi::TypedFunction<Expr(const BlockBuilder& bb, const Call& call)>;

/*! \brief The function type of a function to lower the runtime builtin.
 *
 * A builtin function may be lowered to a lowered form in `LowerRuntimeBuiltin`.
 *
 * \param bb The BlockBuilder context.
 * \param call The call to be lowered.
 */
using FLowerBuiltin = ffi::TypedFunction<Expr(const BlockBuilder& bb, const Call& call)>;

/*!
 * \brief Gradient for a specific op.
 *
 * \param orig_var the original var corresponding to orig_call.
 * \param orig_call the original Call(op) expr.
 * \param output_grad the gradient of the Expr.
 * \param ctx the current block builder context.
 * \return the gradient for each parameter.
 */
using FPrimalGradient = ffi::TypedFunction<tvm::Array<Expr>(
    const Var& orig_var, const Call& orig_call, const Var& output_grad, const BlockBuilder& ctx)>;

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_ATTR_TYPES_H_

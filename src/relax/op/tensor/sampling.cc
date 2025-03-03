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
 * \file sampling.cc
 * \brief sampling operators.
 */

#include "sampling.h"

#include <tvm/relax/analysis.h>

#include <utility>

namespace tvm {
namespace relax {

/* relax.multinomial_from_uniform */
TVM_REGISTER_NODE_TYPE(MultinomialFromUniformAttrs);

Expr multinomial_from_uniform(Expr prob, Expr uniform_sample, Expr sample_indices, DataType dtype) {
  ObjectPtr<MultinomialFromUniformAttrs> attrs = make_object<MultinomialFromUniformAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.multinomial_from_uniform");
  return Call(op, {std::move(prob), std::move(uniform_sample), std::move(sample_indices)},
              Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.multinomial_from_uniform").set_body_typed(multinomial_from_uniform);

StructInfo InferStructInfoMultinomialFromUniform(const Call& call, const BlockBuilder& ctx) {
  CheckNumArguments(call, ctx);
  TensorStructInfo prob_sinfo = GetInputTensorStructInfo(call, 0, ctx);
  TensorStructInfo uniform_sample_sinfo = GetInputTensorStructInfo(call, 1, ctx);
  TensorStructInfo sample_indices_sinfo = GetInputTensorStructInfo(call, 2, ctx);
  const auto* attrs = call->attrs.as<MultinomialFromUniformAttrs>();

  if (!prob_sinfo->dtype.is_float()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Multinomial_from_uniform op requires the input prob to have float dtype. "
                        "However, the given prob dtype is "
                     << prob_sinfo->dtype);
  }
  if (!uniform_sample_sinfo->dtype.is_float()) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Multinomial_from_uniform op requires the input uniform_sample to have float "
           "dtype. However, the given uniform_sample dtype is "
        << uniform_sample_sinfo->dtype);
  }
  if (!sample_indices_sinfo->dtype.is_int()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Multinomial from uniform op requires the input sample_indices to have int "
                        "dtype. However, the given sample_indices dtype is "
                     << sample_indices_sinfo->dtype);
  }
  if (prob_sinfo->IsUnknownNdim() || uniform_sample_sinfo->IsUnknownNdim() ||
      sample_indices_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(attrs->dtype, kUnknownNDim, prob_sinfo->vdevice);
  }
  if (prob_sinfo->ndim != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Multinomial_from_uniform op requires the input prob to be a 2D tensor. "
                        "However, the given prob tensor has ndim "
                     << prob_sinfo->ndim);
  }
  if (uniform_sample_sinfo->ndim != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Multinomial_from_uniform op requires the input uniform_sample to be a 2D "
                        "tensor. However, the given uniform_sample tensor has ndim "
                     << uniform_sample_sinfo->ndim);
  }
  if (sample_indices_sinfo->ndim != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Multinomial_from_uniform op requires the input sample_indices to be a 2D "
                        "tensor. However, the given sample_indices tensor has ndim "
                     << sample_indices_sinfo->ndim);
  }

  // Expected to be `(batch, vocab_size)`
  const auto* prob_shape = prob_sinfo->shape.as<ShapeExprNode>();
  // Expected to be `(n, 1)`
  const auto* uniform_sample_shape = uniform_sample_sinfo->shape.as<ShapeExprNode>();
  // Expected to be `(n, 1)`
  const auto* sample_indices_shape = sample_indices_sinfo->shape.as<ShapeExprNode>();
  // The output shape is expected to be `(n, 1)`

  if (prob_shape == nullptr || uniform_sample_shape == nullptr || sample_indices_shape == nullptr) {
    return TensorStructInfo(attrs->dtype, 2, prob_sinfo->vdevice);
  }

  PrimExpr batch = prob_shape->values[0];
  PrimExpr n = uniform_sample_shape->values[0];
  arith::Analyzer ana;
  if (!ana.CanProveEqual(n, sample_indices_shape->values[0])) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Multinomial_from_uniform op requires the input uniform_sample and "
                        "sample_indices to have the same batch size. "
                        "However, the given uniform_sample tensor has batch size `"
                     << n << "` and the given sample_indices tensor has batch size `"
                     << sample_indices_shape->values[0] << "`");
  }
  if (!tir::is_one(uniform_sample_shape->values[1]) ||
      !tir::is_one(sample_indices_shape->values[1])) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Multinomial_from_uniform op requires the input uniform_sample and "
                        "sample_indices to be 2D tensors with the second dimension being 1. "
                        "However, the given uniform_sample tensor has shape "
                     << uniform_sample_sinfo->shape
                     << " and the given sample_indices tensor has shape "
                     << sample_indices_sinfo->shape);
  }
  return TensorStructInfo(ShapeExpr({n, 1}), attrs->dtype, prob_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.multinomial_from_uniform")
    .set_attrs_type<MultinomialFromUniformAttrs>()
    .set_num_inputs(3)
    .add_argument("prob", "Tensor", "The probability tensor.")
    .add_argument("uniform_sample", "Tensor", "The uniform sample tensor.")
    .add_argument("sample_indices", "Tensor", "The sample indices tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMultinomialFromUniform)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm

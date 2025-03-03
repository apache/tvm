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

#include "attention.h"

#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.nn.attention */
TVM_REGISTER_NODE_TYPE(AttentionAttrs);

Expr attention(Expr query, Expr key, Expr value, Optional<Expr> bias, Optional<FloatImm> scale,
               Optional<String> causal_mask, Optional<IntImm> window_size) {
  ObjectPtr<AttentionAttrs> attrs = make_object<AttentionAttrs>();
  attrs->scale = scale;
  attrs->causal_mask = causal_mask;
  attrs->window_size = window_size;

  if (bias) {
    return Call(Op::Get("relax.nn.attention_bias"),
                {std::move(query), std::move(key), std::move(value), std::move(bias.value())},
                Attrs(attrs), {});
  }
  return Call(Op::Get("relax.nn.attention"), {std::move(query), std::move(key), std::move(value)},
              Attrs(attrs), {});
}

Expr attention_var_len(Expr query, Expr key, Expr value, Expr seqstart_q, Expr seqstart_k,
                       Expr max_seqlen_q, Expr max_seqlen_k, Optional<FloatImm> scale,
                       Optional<String> causal_mask, Optional<IntImm> window_size) {
  ObjectPtr<AttentionAttrs> attrs = make_object<AttentionAttrs>();
  attrs->scale = scale;
  attrs->causal_mask = causal_mask;
  attrs->window_size = window_size;

  return Call(Op::Get("relax.nn.attention_var_len"),
              {query, key, value, seqstart_q, seqstart_k, max_seqlen_q, max_seqlen_k}, Attrs(attrs),
              {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.attention").set_body_typed(attention);
TVM_REGISTER_GLOBAL("relax.op.nn.attention_var_len").set_body_typed(attention_var_len);

StructInfo InferStructInfoAttention(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo q_sinfo = input_sinfo[0];
  TensorStructInfo k_sinfo = input_sinfo[1];
  TensorStructInfo v_sinfo = input_sinfo[2];
  auto diag_dim = [&](TensorStructInfo sinfo, String name) {
    if (sinfo->ndim != 4) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "The " << name << " should have 4 dimension, namely "
                       << "[batch size, sequence length, number of heads, dimension of heads].");
    }
  };
  diag_dim(q_sinfo, "query");
  diag_dim(k_sinfo, "key");
  diag_dim(v_sinfo, "value");
  const ShapeExprNode* q_shape = q_sinfo->shape.as<ShapeExprNode>();
  const ShapeExprNode* k_shape = k_sinfo->shape.as<ShapeExprNode>();
  const ShapeExprNode* v_shape = v_sinfo->shape.as<ShapeExprNode>();
  PrimExpr num_batches = q_shape->values[0];
  PrimExpr num_queries = q_shape->values[1];
  PrimExpr num_heads = q_shape->values[2];
  PrimExpr head_dim = q_shape->values[3];
  PrimExpr num_keys = k_shape->values[1];
  PrimExpr head_dim_value = v_shape->values[3];
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  auto diag_equal = [&](PrimExpr v1, PrimExpr v2, String m1, String m2, String dim) {
    if (analyzer->CanProve(v1 != v2)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "The " << m1 << " " << dim << " and the " << m2 << " " << dim
                       << " should be the same. However, the " << dim << " of " << m1 << " is "
                       << v1 << " while the " << dim << " of " << m2 << " is " << v2);
    }
  };
  auto multiple_of = [&](PrimExpr v1, PrimExpr v2, String m1, String m2, String dim) {
    if (analyzer->CanProve(indexmod(v1, v2) != 0)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "The " << m1 << " " << dim << " should be a multiple of " << m2 << " "
                       << dim << ". However, the " << dim << " of " << m1 << " is " << v1
                       << " while the " << dim << " of " << m2 << " is " << v2);
    }
  };

  diag_equal(num_batches, k_shape->values[0], "query", "key", "batch size");
  diag_equal(num_batches, v_shape->values[0], "query", "value", "batch size");
  multiple_of(num_heads, k_shape->values[2], "query", "key", "number of heads");
  multiple_of(num_heads, v_shape->values[2], "query", "value", "number of heads");
  diag_equal(num_keys, v_shape->values[1], "key", "value", "sequence length");
  diag_equal(head_dim, k_shape->values[3], "query", "key", "dimension of heads");

  if (input_sinfo.size() == 4) {
    TensorStructInfo bias_sinfo = input_sinfo[3];
    const ShapeExprNode* bias_shape = bias_sinfo->shape.as<ShapeExprNode>();
    if (bias_sinfo->ndim != 4) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "The bias should have 4 dimensions."
                       << "However, the bias input has " << bias_sinfo->ndim << " dimensions.");
    }
    auto diag_equal_or_broadcast = [&](PrimExpr v1, PrimExpr v2, String m1, String m2, String dim) {
      if (analyzer->CanProve(v1 != v2) && !tir::is_one(v2)) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "The " << m1 << " " << dim << " and the " << m2 << " " << dim
                         << " should be the same or broadcastable. However, the " << dim << " of "
                         << m1 << " is " << v1 << " while the " << dim << " of " << m2 << " is "
                         << v2);
      }
    };
    diag_equal_or_broadcast(num_batches, bias_shape->values[0], "query", "bias", "batch size");
    diag_equal_or_broadcast(num_heads, bias_shape->values[1], "query", "bias", "number of heads");
    diag_equal_or_broadcast(num_queries, bias_shape->values[2], "query", "bias", "sequence length");
    diag_equal(num_keys, bias_shape->values[3], "key", "bias", "sequence length");
  }

  Array<PrimExpr> output_shape = {num_batches, num_queries, num_heads, head_dim_value};
  return TensorStructInfo(ShapeExpr(output_shape), q_sinfo->dtype, q_sinfo->vdevice);
}

Call InferMixedPrecisionAttention(const Call& call, const DataType& out_dtype) {
  return Downcast<Call>(
      attention(call->args[0], call->args[1], call->args[2], NullOpt, NullOpt, NullOpt, NullOpt));
}

TVM_REGISTER_OP("relax.nn.attention")
    .set_attrs_type<AttentionAttrs>()
    .set_num_inputs(3)
    .add_argument("query", "Tensor", "The input queries tensor.")
    .add_argument("key", "Tensor", "The input keys tensor.")
    .add_argument("value", "Tensor", "The input values tensor.")
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionAttention)
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAttention)
    .set_attr<Bool>("FPurity", Bool(true));

TVM_REGISTER_OP("relax.nn.attention_bias")
    .set_attrs_type<AttentionAttrs>()
    .set_num_inputs(4)
    .add_argument("query", "Tensor", "The input queries tensor.")
    .add_argument("key", "Tensor", "The input keys tensor.")
    .add_argument("value", "Tensor", "The input values tensor.")
    .add_argument("bias", "Tensor", "The input bias tensor.")
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionAttention)
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAttention)
    .set_attr<Bool>("FPurity", Bool(true));

TVM_REGISTER_OP("relax.nn.attention_var_len")
    .set_attrs_type<AttentionAttrs>()
    .set_num_inputs(7)
    .add_argument("query", "Tensor", "The input queries tensor.")
    .add_argument("key", "Tensor", "The input keys tensor.")
    .add_argument("value", "Tensor", "The input values tensor.")
    .add_argument("seqstart_q", "Tensor", "The cumsum of query sequence lengths, prepended with 0.")
    .add_argument("seqstart_k", "Tensor", "The cumsum of key sequence lengths, prepended with 0.")
    .add_argument("max_seqlen_q", "Tensor", "The maximum query sequence length in the batch.")
    .add_argument("max_seqlen_k", "Tensor", "The maximum key sequence length in the batch.")
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionAttention)
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAttention)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm

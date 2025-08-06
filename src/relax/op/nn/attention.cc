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

#include <tvm/ffi/reflection/registry.h>

#include <utility>

namespace tvm {
namespace relax {

/* relax.nn.attention */

Expr attention(Expr query, Expr key, Expr value, Optional<Expr> bias, Optional<FloatImm> scale,
               Optional<String> causal_mask, Optional<IntImm> window_size,
               Optional<bool> enable_gqa, Optional<IntImm> num_kv_heads) {
  ObjectPtr<AttentionAttrs> attrs = make_object<AttentionAttrs>();
  attrs->scale = scale;
  attrs->causal_mask = causal_mask;
  attrs->window_size = window_size;

  attrs->enable_gqa = enable_gqa;
  attrs->num_kv_heads = num_kv_heads;
  if (bias) {
    return Call(Op::Get("relax.nn.attention"),
                {std::move(query), std::move(key), std::move(value), bias.value()}, Attrs(attrs),
                {});
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

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.op.nn.attention", attention)
      .def("relax.op.nn.attention_var_len", attention_var_len);
});

StructInfo InferStructInfoAttention(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo q_sinfo = input_sinfo[0];
  TensorStructInfo k_sinfo = input_sinfo[1];
  TensorStructInfo v_sinfo = input_sinfo[2];

  auto check_4d = [&](TensorStructInfo sinfo, const String& name) {
    if (sinfo->ndim != 4) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "The " << name
                       << " tensor must be 4D with shape [B, S, H, D]. Got ndim = " << sinfo->ndim);
    }
  };

  check_4d(q_sinfo, "query");
  check_4d(k_sinfo, "key");
  check_4d(v_sinfo, "value");

  const ShapeExprNode* q_shape = q_sinfo->shape.as<ShapeExprNode>();
  const ShapeExprNode* k_shape = k_sinfo->shape.as<ShapeExprNode>();
  const ShapeExprNode* v_shape = v_sinfo->shape.as<ShapeExprNode>();

  PrimExpr B = q_shape->values[0];
  PrimExpr S_q = q_shape->values[1];
  PrimExpr H_q = q_shape->values[2];
  PrimExpr D_q = q_shape->values[3];

  PrimExpr S_k = k_shape->values[1];
  PrimExpr H_k = k_shape->values[2];
  PrimExpr D_k = k_shape->values[3];

  PrimExpr S_v = v_shape->values[1];
  PrimExpr H_v = v_shape->values[2];
  PrimExpr D_v = v_shape->values[3];

  arith::Analyzer* analyzer = ctx->GetAnalyzer();

  auto assert_equal = [&](PrimExpr a, PrimExpr b, const String& a_name, const String& b_name,
                          const String& dim) {
    if (analyzer->CanProve(a != b)) {
      ctx->ReportFatal(Diagnostic::Error(call) << "Mismatch in " << dim << ": " << a_name << " has "
                                               << a << ", but " << b_name << " has " << b << ".");
    }
  };

  auto assert_multiple_of = [&](PrimExpr a, PrimExpr b, const String& a_name, const String& b_name,
                                const String& dim) {
    if (analyzer->CanProve(indexmod(a, b) != 0)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << a_name << " " << dim << " must be multiple of " << b_name << " " << dim
                       << ". Got " << a << " vs " << b);
    }
  };

  // Check batch size
  assert_equal(B, k_shape->values[0], "query", "key", "batch size");
  assert_equal(B, v_shape->values[0], "query", "value", "batch size");

  // Check head dim
  assert_equal(D_q, D_k, "query", "key", "head dim");

  // Check sequence match
  assert_equal(S_k, S_v, "key", "value", "sequence length");

  const auto* attrs = call->attrs.as<AttentionAttrs>();
  if (!attrs) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Missing AttentionAttrs.");
  }

  if (attrs->enable_gqa.value_or(false)) {
    PrimExpr H_kv = H_k;
    PrimExpr H_qv = H_q;

    // Ensure query head count is multiple of key/value head count
    assert_multiple_of(H_qv, H_kv, "query", "key/value", "head count");

    // Value head count must also match key for GQA
    assert_equal(H_k, H_v, "key", "value", "head count");

    // Optional: warn if num_kv_heads != H_k (could be runtime config mismatch)
    if (attrs->num_kv_heads) {
      auto kv_heads = attrs->num_kv_heads.value();
      if (analyzer->CanProve(H_k != kv_heads)) {
        LOG(WARNING) << "Attention GQA: num_kv_heads attribute (" << kv_heads
                     << ") doesn't match key head count (" << H_k << ").";
      }
    }
  } else {
    // Standard MHA: check head count matches
    assert_equal(H_q, H_k, "query", "key", "head count");
    assert_equal(H_q, H_v, "query", "value", "head count");
  }

  // Optional bias check
  if (input_sinfo.size() == 4) {
    TensorStructInfo bias_sinfo = input_sinfo[3];
    if (bias_sinfo->ndim != 4) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Bias tensor must be 4D. Got ndim = " << bias_sinfo->ndim);
    }
    const ShapeExprNode* bias_shape = bias_sinfo->shape.as<ShapeExprNode>();

    auto assert_broadcastable = [&](PrimExpr val, PrimExpr bias_val, const String& name) {
      if (analyzer->CanProve(val != bias_val) && !tir::is_one(bias_val)) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << name << " mismatch in bias. Got " << val << " vs " << bias_val);
      }
    };

    assert_broadcastable(B, bias_shape->values[0], "batch size");
    assert_broadcastable(H_q, bias_shape->values[1], "num_heads");
    assert_broadcastable(S_q, bias_shape->values[2], "query sequence length");
    assert_broadcastable(S_k, bias_shape->values[3], "key sequence length");
  }

  Array<PrimExpr> out_shape = {B, S_q, H_q, D_v};
  return TensorStructInfo(ShapeExpr(out_shape), q_sinfo->dtype, q_sinfo->vdevice);
}

// StructInfo InferStructInfoAttention(const Call& call, const BlockBuilder& ctx) {
//   Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
//   TensorStructInfo q_sinfo = input_sinfo[0];
//   TensorStructInfo k_sinfo = input_sinfo[1];
//   TensorStructInfo v_sinfo = input_sinfo[2];
//   auto diag_dim = [&](TensorStructInfo sinfo, String name) {
//     if (sinfo->ndim != 4) {
//       ctx->ReportFatal(Diagnostic::Error(call)
//                        << "The " << name << " should have 4 dimension, namely "
//                        << "[batch size, sequence length, number of heads, dimension of heads].");
//     }
//   };
//   diag_dim(q_sinfo, "query");
//   diag_dim(k_sinfo, "key");
//   diag_dim(v_sinfo, "value");
//   const ShapeExprNode* q_shape = q_sinfo->shape.as<ShapeExprNode>();
//   const ShapeExprNode* k_shape = k_sinfo->shape.as<ShapeExprNode>();
//   const ShapeExprNode* v_shape = v_sinfo->shape.as<ShapeExprNode>();
//   PrimExpr num_batches = q_shape->values[0];
//   PrimExpr num_queries = q_shape->values[1];
//   PrimExpr num_heads = q_shape->values[2];
//   PrimExpr head_dim = q_shape->values[3];
//   PrimExpr num_keys = k_shape->values[1];
//   PrimExpr head_dim_value = v_shape->values[3];
//   arith::Analyzer* analyzer = ctx->GetAnalyzer();
//   auto diag_equal = [&](PrimExpr v1, PrimExpr v2, String m1, String m2, String dim) {
//     if (analyzer->CanProve(v1 != v2)) {
//       ctx->ReportFatal(Diagnostic::Error(call)
//                        << "The " << m1 << " " << dim << " and the " << m2 << " " << dim
//                        << " should be the same. However, the " << dim << " of " << m1 << " is "
//                        << v1 << " while the " << dim << " of " << m2 << " is " << v2);
//     }
//   };
//   auto multiple_of = [&](PrimExpr v1, PrimExpr v2, String m1, String m2, String dim) {
//     if (analyzer->CanProve(indexmod(v1, v2) != 0)) {
//       ctx->ReportFatal(Diagnostic::Error(call)
//                        << "The " << m1 << " " << dim << " should be a multiple of " << m2 << " "
//                        << dim << ". However, the " << dim << " of " << m1 << " is " << v1
//                        << " while the " << dim << " of " << m2 << " is " << v2);
//     }
//   };

//   diag_equal(num_batches, k_shape->values[0], "query", "key", "batch size");
//   diag_equal(num_batches, v_shape->values[0], "query", "value", "batch size");
//   multiple_of(num_heads, k_shape->values[2], "query", "key", "number of heads");
//   multiple_of(num_heads, v_shape->values[2], "query", "value", "number of heads");
//   diag_equal(num_keys, v_shape->values[1], "key", "value", "sequence length");
//   diag_equal(head_dim, k_shape->values[3], "query", "key", "dimension of heads");

//   if (input_sinfo.size() == 4) {
//     TensorStructInfo bias_sinfo = input_sinfo[3];
//     const ShapeExprNode* bias_shape = bias_sinfo->shape.as<ShapeExprNode>();
//     if (bias_sinfo->ndim != 4) {
//       ctx->ReportFatal(Diagnostic::Error(call)
//                        << "The bias should have 4 dimensions."
//                        << "However, the bias input has " << bias_sinfo->ndim << " dimensions.");
//     }
//     auto diag_equal_or_broadcast = [&](PrimExpr v1, PrimExpr v2, String m1, String m2, String
//     dim) {
//       if (analyzer->CanProve(v1 != v2) && !tir::is_one(v2)) {
//         ctx->ReportFatal(Diagnostic::Error(call)
//                          << "The " << m1 << " " << dim << " and the " << m2 << " " << dim
//                          << " should be the same or broadcastable. However, the " << dim << " of
//                          "
//                          << m1 << " is " << v1 << " while the " << dim << " of " << m2 << " is "
//                          << v2);
//       }
//     };
//     diag_equal_or_broadcast(num_batches, bias_shape->values[0], "query", "bias", "batch size");
//     diag_equal_or_broadcast(num_heads, bias_shape->values[1], "query", "bias", "number of
//     heads"); diag_equal_or_broadcast(num_queries, bias_shape->values[2], "query", "bias",
//     "sequence length"); diag_equal(num_keys, bias_shape->values[3], "key", "bias", "sequence
//     length");
//   }
//   Array<PrimExpr> output_shape = {num_batches, num_queries, num_heads, head_dim_value};
//   return TensorStructInfo(ShapeExpr(output_shape), q_sinfo->dtype, q_sinfo->vdevice);
// }

Call InferMixedPrecisionAttention(const Call& call, const DataType& out_dtype) {
  return Downcast<Call>(attention(call->args[0], call->args[1], call->args[2], std::nullopt,
                                  std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                                  std::nullopt));
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

TVM_FFI_STATIC_INIT_BLOCK({ AttentionAttrs::RegisterReflection(); });

}  // namespace relax
}  // namespace tvm

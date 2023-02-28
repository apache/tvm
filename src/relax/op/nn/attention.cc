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

Expr attention(Expr query, Expr key, Expr value, DataType out_dtype) {
  ObjectPtr<AttentionAttrs> attrs = make_object<AttentionAttrs>();
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.nn.attention");
  return Call(op, {std::move(query), std::move(key), std::move(value)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.attention").set_body_typed(attention);

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
  const auto* attrs = call->attrs.as<AttentionAttrs>();
  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, q_sinfo, k_sinfo)
                           : attrs->out_dtype;
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
  diag_equal(num_batches, k_shape->values[0], "query", "key", "batch size");
  diag_equal(num_batches, v_shape->values[0], "query", "value", "batch size");
  diag_equal(num_heads, k_shape->values[2], "query", "key", "number of heads");
  diag_equal(num_heads, v_shape->values[2], "query", "value", "number of heads");
  diag_equal(k_shape->values[1], v_shape->values[1], "key", "value", "sequence length");
  diag_equal(head_dim, k_shape->values[3], "query", "key", "dimension of heads");

  Array<PrimExpr> output_shape = {num_batches, num_queries, num_heads, head_dim_value};
  return TensorStructInfo(ShapeExpr(output_shape), out_dtype);
}

TVM_REGISTER_OP("relax.nn.attention")
    .set_num_inputs(3)
    .add_argument("query", "Tensor", "The input queries tensor.")
    .add_argument("key", "Tensor", "The input keys tensor.")
    .add_argument("value", "Tensor", "The input values tensor.")
    .set_attrs_type<AttentionAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAttention);

}  // namespace relax
}  // namespace tvm

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
 * \file manipulate.cc
 * \brief Manipulation operators.
 */

#include "manipulate.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

// Helper function for flatten and reshape.
PrimExpr ComputeShapeProduct(const Array<PrimExpr>& shape_values) {
  PrimExpr shape_prod = IntImm(DataType::Int(64), 1);
  for (PrimExpr value : shape_values) {
    shape_prod *= value;
  }
  return shape_prod;
}

/* relax.reshape */
Expr ConvertNewShapeToExpr(const Expr& data, const ObjectRef& shape) {
  if (const auto* e = shape.as<ExprNode>()) {
    return GetRef<Expr>(e);
  }

  const auto* array = shape.as<ArrayNode>();
  CHECK(array != nullptr) << "Reshape only expects the input new shape to be either an Expr or an "
                             "Array of PrimExprs. However, the given new shape is "
                          << shape;
  int dim_to_infer = -1;
  PrimExpr new_shape_prod = IntImm(DataType::Int(64), 1);
  for (int i = 0; i < static_cast<int>(array->size()); ++i) {
    const auto* _len = array->at(i).as<PrimExprNode>();
    CHECK(_len != nullptr) << "Reshape only expects the input new shape to be either an Expr or an "
                              "Array of PrimExprs. However, the given new shape is "
                           << shape;
    PrimExpr len = GetRef<PrimExpr>(_len);
    CHECK(len->dtype.is_int()) << "Reshape requires the new shape values to be all "
                                  "integers. However, the give new shape is "
                               << shape;
    const auto* int_len = len.as<IntImmNode>();
    if (int_len != nullptr && int_len->value == -1) {
      CHECK_EQ(dim_to_infer, -1) << "Reshape accepts at most one \"-1\" in the new shape. However, "
                                    "there are multiple \"-1\" in the given new shape  "
                                 << shape;
      dim_to_infer = i;
    } else {
      CHECK(int_len == nullptr || int_len->value > 0)
          << "Reshape requires all values in the new shape to be positive except a single \"-1\". "
             "However, the given new shape is "
          << shape;
      // We expect any symbolic not to signal the intent of -1, and therefore do no check for
      // symbolic value here.
      new_shape_prod = new_shape_prod * len;
    }
  }

  Array<PrimExpr> array_ref = GetRef<Array<PrimExpr>>(array);
  // When there is no dimension to infer, just return the input array as ShapeExpr.
  if (dim_to_infer == -1) {
    return ShapeExpr(array_ref);
  }

  // Otherwise, we require the input tensor to have known shape value for inference.
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(data);
  CHECK(data_sinfo != nullptr)
      << "Reshape expects the input data to be a Tensor. However, the given input is "
      << data->struct_info_->GetTypeKey();
  CHECK(data_sinfo->shape.defined())
      << "Reshape expects the input tensor to have known shape when there is some dimension length "
         "to infer. However, the given input has no shape.";
  const auto* shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(data_sinfo->shape.value());
  CHECK(shape_sinfo != nullptr && shape_sinfo->values.defined())
      << "Reshape expects the input tensor to have known shape when there is some dimension length "
         "to infer. However, the given input shape is "
      << data_sinfo->shape << " whose shape value is unknown.";

  arith::Analyzer analyzer;
  PrimExpr old_shape_prod = ComputeShapeProduct(shape_sinfo->values.value());
  array_ref.Set(dim_to_infer, analyzer.Simplify(floordiv(old_shape_prod, new_shape_prod)));
  return ShapeExpr(array_ref);
}

Expr reshape(Expr x, ObjectRef shape) {
  Expr shape_in_expr = ConvertNewShapeToExpr(x, shape);
  static const Op& op = Op::Get("relax.reshape");
  return Call(op, {std::move(x), std::move(shape_in_expr)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.reshape").set_body_typed(reshape);

StructInfo InferStructInfoReshape(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call->span) << "Reshape op should take 2 arguments");
  }
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* new_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);
  if (data_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call->span)
                     << "Reshape requires the input data to be Tensor. However, the given one is "
                     << call->args[0]->struct_info_->GetTypeKey());
  }
  if (new_shape_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call->span)
        << "Reshape requires the input new shape to be Shape. However, the given one is "
        << call->args[1]->struct_info_->GetTypeKey());
  }

  Optional<Array<PrimExpr>> old_shape_values;
  if (data_sinfo->shape.defined()) {
    const auto* old_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(data_sinfo->shape.value());
    ICHECK_NOTNULL(old_shape_sinfo);
    old_shape_values = old_shape_sinfo->values;
  }

  if (new_shape_sinfo->values.defined() && old_shape_values.defined()) {
    PrimExpr new_shape_prod = ComputeShapeProduct(new_shape_sinfo->values.value());
    PrimExpr old_shape_prod = ComputeShapeProduct(old_shape_values.value());
    if (ctx->GetAnalyzer()->CanProve(old_shape_prod != new_shape_prod)) {
      ctx->ReportFatal(Diagnostic::Error(call->span)
                       << "Reshape expects the new shape to be convertible from the old shape. "
                          "However, the old shape is "
                       << data_sinfo->shape << ", with product " << old_shape_prod
                       << ", while the new shape is " << call->args[1] << ", with product "
                       << new_shape_prod);
    }
  }
  return TensorStructInfo(call->args[1], data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.reshape")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("shape", "Shape", "The input new shape.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReshape);

}  // namespace relax
}  // namespace tvm

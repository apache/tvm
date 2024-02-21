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
 * \file relax/src/ir/emit_te.cc
 */
#include "./emit_te.h"

#include <tvm/relax/struct_info.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace relax {

// RXPlaceholderOpNode
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RXPlaceholderOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const RXPlaceholderOpNode*>(node.get());
      p->stream << "rxplaceholder(" << op->name << ", " << op << ")";
    });

TVM_REGISTER_NODE_TYPE(RXPlaceholderOpNode);

te::Tensor TETensor(Expr value, Map<tir::Var, PrimExpr> tir_var_map, std::string name) {
  auto n = make_object<RXPlaceholderOpNode>();
  n->name = name;
  n->value = value;

  // If the value is a constant, it might come as an argument of EmitTE and thus its shape and
  // checked-type might not be properly set. In this case we set the shape and dtype of the returned
  // TE tensor.
  if (const auto* constant = value.as<ConstantNode>()) {
    n->dtype = DataType(constant->data->dtype);

    int ndim = constant->data->ndim;
    ShapeTuple shape_tuple = constant->data.Shape();
    Array<PrimExpr> shape;
    shape.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      shape.push_back(IntImm(DataType::Int(64), shape_tuple[i]));
    }
    n->shape = std::move(shape);
    return te::PlaceholderOp(n).output(0);
  }
  ICHECK(value->struct_info_.defined()) << "value must be normalized and contain StructInfo";
  auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(value);
  ICHECK(tensor_sinfo) << "Value must be a tensor";
  auto* shape_expr = tensor_sinfo->shape.as<ShapeExprNode>();
  CHECK(shape_expr)
      << "ValueError: Expression does not have an known symbolic shape, please consider use "
         "match_cast "
      << "to constrain the shape before passing into te_tensor";
  n->shape = shape_expr->values.Map(
      [&tir_var_map](const PrimExpr& e) { return tir::Substitute(e, tir_var_map); });
  n->dtype = tensor_sinfo->dtype;
  return te::PlaceholderOp(n).output(0);
}

TVM_REGISTER_GLOBAL("relax.TETensor").set_body_typed(TETensor);

}  // namespace relax
}  // namespace tvm

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
 * \file tvm/relax/transform/legalize_ops.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

/*!
 * \brief Check if a given Tensor/Shape/TupleStructInfo contains shapes whose
 * values are all known.
 * \param sinfo The StructInfo to be checked.
 * \return A boolean indicating the given struct info contains shape values that are all known.
 */
bool KnowAllShapeValues(const StructInfo& sinfo) {
  if (const auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
    return tensor_sinfo->shape.defined() &&
           tensor_sinfo->shape.value()->IsInstance<ShapeExprNode>();
  } else if (const auto* shape_sinfo = sinfo.as<ShapeStructInfoNode>()) {
    return shape_sinfo->values.defined();
  } else if (const auto* tuple_sinfo = sinfo.as<TupleStructInfoNode>()) {
    return std::all_of(tuple_sinfo->fields.begin(), tuple_sinfo->fields.end(),
                       [](StructInfo field_sinfo) { return KnowAllShapeValues(field_sinfo); });
  } else if (sinfo.as<PrimStructInfoNode>()) {
    return true;
  } else {
    return false;
  }
}

class LegalizeMutator : public ExprMutator {
 public:
  explicit LegalizeMutator(const IRModule& mod, const Optional<Map<String, PackedFunc>>& cmap)
      : ExprMutator(mod), mod_(std::move(mod)), cmap_(std::move(cmap)) {}

  IRModule Transform() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<FunctionNode>()) {
        auto updated_func = Downcast<Function>(this->VisitExpr(func));
        builder_->UpdateFunction(gv, Downcast<BaseFunc>(updated_func));
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call) final {
    Call visited_call = Downcast<Call>(this->VisitExprPostOrder_(call));
    static const auto& legalize_map = Op::GetAttrMap<FLegalize>("FLegalize");
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    static const Op& call_dps_packed_op = Op::Get("relax.call_dps_packed");
    auto* op_node = visited_call->op.as<OpNode>();

    // Not an OpNode
    if (op_node == nullptr) {
      return visited_call;
    }

    // Not all shape values are known
    if (!std::all_of(visited_call->args.begin(), visited_call->args.end(),
                     [](Expr arg) { return KnowAllShapeValues(GetStructInfo(arg)); }) ||
        !KnowAllShapeValues(GetStructInfo(visited_call))) {
      return visited_call;
    }

    auto op = GetRef<Op>(op_node);

    // Priority: customize > default.
    // Check if it has customize legalization registered.
    if (cmap_.defined() && cmap_.value().count(op->name)) {
      return cmap_.value()[op->name](this->builder_, visited_call);
    }
    // Check if it has default legalization registered.
    if (legalize_map.count(op)) {
      return legalize_map[op](this->builder_, visited_call);
    }

    // No legalization.
    if (op != call_tir_op && op != call_dps_packed_op) {
      LOG(WARNING) << "No legalization func for " << op->name << " is found.";
    }
    return visited_call;
  }

  /*! \brief The context IRModule. */
  IRModule mod_;
  /*! \brief The customized legalization function map. */
  Optional<Map<String, PackedFunc>> cmap_;
};

namespace transform {

Pass LegalizeOps(Optional<Map<String, PackedFunc>> cmap) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return LegalizeMutator(mod, cmap).Transform(); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"LegalizeOps",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.LegalizeOps").set_body_typed(LegalizeOps);

}  // namespace transform

}  // namespace relax
}  // namespace tvm

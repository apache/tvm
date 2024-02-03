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
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

TVM_REGISTER_PASS_CONFIG_OPTION("relax.transform.apply_legalize_ops", Bool);

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
  explicit LegalizeMutator(const IRModule& mod, const Optional<Map<String, PackedFunc>>& cmap,
                           bool enable_warning)
      : ExprMutator(mod), mod_(std::move(mod)), enable_warning_(enable_warning) {
    if (cmap) {
      cmap_ = std::move(cmap.value());
    }
  }

  IRModule Transform() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<FunctionNode>()) {
        auto updated_func = Downcast<Function>(this->VisitExpr(func));
        builder_->UpdateFunction(gv, Downcast<BaseFunc>(updated_func));
      }
    }
    // Fill the "kTarget" attribute of PrimFunc
    for (const auto& [gv, func] : builder_->GetContextIRModule()->functions) {
      const tir::PrimFuncNode* prim_func;
      if (tmap_.count(gv) && (prim_func = func.as<tir::PrimFuncNode>())) {
        auto f = WithAttr(GetRef<tir::PrimFunc>(prim_func), tvm::attr::kTarget, tmap_[gv]);
        builder_->UpdateFunction(gv, f);
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  bool WrapPureCondition(const Op& op, const Expr& legalized) {
    static const auto& purity_map = Op::GetAttrMap<Bool>("FPurity");

    // unlikely for this condition not to be met
    if (const CallNode* call = legalized.as<CallNode>()) {
      // if the original op is not pure, don't wrap
      if (!(purity_map.count(op) && purity_map[op]->value)) {
        return false;
      }
      if (const OpNode* call_op = call->op.as<OpNode>()) {
        auto res_op = GetRef<Op>(call_op);
        if (purity_map.count(res_op)) {
          // if the legalized op is already pure, we *don't* need a wrapper
          return !purity_map[res_op]->value;
        }
      }
      // simplest case: wrap if the original op was pure and the result is somehow not
      return true;
    }
    return false;
  }

  Call WrapPureCall(const Call& ret) {
    static const Op& call_pure_packed_op = Op::Get("relax.call_pure_packed");
    Array<Expr> ret_args = {ret->op};
    for (auto arg : ret->args) {
      ret_args.push_back(arg);
    }
    return Call(call_pure_packed_op, ret_args, ret->attrs, ret->sinfo_args);
  }

  Target GetTarget(const Array<StructInfo>& sinfos) {
    for (auto sinfo : sinfos) {
      if (const auto* tinfo = sinfo.as<TensorStructInfoNode>()) {
        if (tinfo->vdevice.defined()) {
          auto vdevice = tinfo->vdevice.value();
          if (vdevice->target.defined()) {
            return vdevice->target;
          }
        }
      } else if (const auto* tup_sinfo = sinfo.as<TupleStructInfoNode>()) {
        return GetTarget(tup_sinfo->fields);
      }
    }
    return Target();
  }

  void SaveTarget(const Expr& expr) {
    if (expr->IsInstance<CallNode>()) {
      auto call = Downcast<Call>(expr);
      auto target = GetTarget(call->sinfo_args);
      const GlobalVarNode* gvar_node;
      if (target.defined() && (gvar_node = call->args[0].as<GlobalVarNode>())) {
        this->tmap_.Set(GetRef<GlobalVar>(gvar_node), target);
      }
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    Call visited_call = Downcast<Call>(this->VisitExprPostOrder_(call));
    static const auto& legalize_map = Op::GetAttrMap<FLegalize>("FLegalize");
    static const Op& call_pure_packed_op = Op::Get("relax.call_pure_packed");
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    static const Op& call_dps_packed_op = Op::Get("relax.call_dps_packed");
    auto* op_node = visited_call->op.as<OpNode>();

    // Not an OpNode
    if (op_node == nullptr) {
      return visited_call;
    }

    auto op = GetRef<Op>(op_node);
    std::string op_name(op->name);
    bool is_data_dependent_op = (op_name.find("dynamic") != std::string::npos);
    // Not all shape values are known
    // Data-dependent ops are exception since their output shape will be identified at runtime.
    // Legalizer will insert their shape functions, which are manually registered, and match cast
    // to define symbolic output shape at compile time.
    if (!std::all_of(visited_call->args.begin(), visited_call->args.end(),
                     [](Expr arg) { return KnowAllShapeValues(GetStructInfo(arg)); }) ||
        (!is_data_dependent_op && !KnowAllShapeValues(GetStructInfo(visited_call)))) {
      return visited_call;
    }

    FLegalize legalization_func;

    if (auto opt_custom_legalize = cmap_.Get(op->name)) {
      // First choice, use a custom legalization function
      legalization_func = opt_custom_legalize.value();
    } else if (legalize_map.count(op)) {
      // Second choice, use a default legalization
      legalization_func = legalize_map[op];
    } else {
      // No legalization.
      if (enable_warning_ && op != call_tir_op && op != call_dps_packed_op &&
          op != call_pure_packed_op) {
        LOG(WARNING) << "No legalization func for " << op->name << " is found.";
      }
      return visited_call;
    }

    // The legalization function may call `builder_->Emit()` as part
    // of its implementation.  In that case, any operations it emits
    // must be caught such that they be checked for recursive
    // legalization.  This is done by wrapping the legalized value in
    // a SeqExpr, which can first be visited, then unwrapped by the
    // normalization.
    if (builder_->CurrentBlockIsDataFlow()) {
      builder_->BeginDataflowBlock();
    } else {
      builder_->BeginBindingBlock();
    }
    Expr legalized = legalization_func(builder_, visited_call);

    // Save the expected target info. into tmap_
    SaveTarget(legalized);

    legalized = builder_->Normalize(legalized);

    BindingBlock prologue = builder_->EndBlock();
    for (const auto& binding : prologue->bindings) {
      VisitBinding(binding);
    }

    if (WrapPureCondition(op, legalized)) {
      legalized = WrapPureCall(Downcast<Call>(legalized));
    }

    // Legalization may have introduced additional operations that
    // must be legalized as well.  For example, a user-custom
    // intrinsic whose legalization is implemented in terms of relax
    // intrinsics.  The base case of the recursion occurs when no
    // additional legalization steps are found.
    //
    // Only perform recursive legalization when the legalization
    // function returned a modified expression, as some legalizations
    // return the original expression if they are unable to produce a
    // legalized version.
    if (!legalized.same_as(visited_call)) {
      legalized = VisitExpr(legalized);
    }

    return legalized;
  }

  /*! \brief The context IRModule. */
  IRModule mod_;
  /*! \brief The customized legalization function map. */
  Map<String, PackedFunc> cmap_;
  /*! \brief The map from GlobalVar of PrimFunc to compilation Target. */
  Map<GlobalVar, Target> tmap_;
  /*!
   * \brief A boolean value indicating if to print warnings for CallNode whose op's
   * legalization function is not registered.
   */
  bool enable_warning_;
};

namespace transform {

Pass LegalizeOps(Optional<Map<String, PackedFunc>> cmap, bool enable_warning) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext pc) {
    bool apply_legalize_ops =
        pc->GetConfig<Bool>("relax.transform.apply_legalize_ops").value_or(Bool(true))->value;
    if (apply_legalize_ops) {
      mod = LegalizeMutator(mod, cmap, enable_warning).Transform();
    }
    return mod;
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"LegalizeOps",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.LegalizeOps").set_body_typed(LegalizeOps);

}  // namespace transform

}  // namespace relax
}  // namespace tvm

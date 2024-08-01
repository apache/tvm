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
#include <tvm/tir/transform.h>

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
    for (const auto& gv : mod_->GetGlobalVars()) {
      const auto& func = mod_->Lookup(gv);
      if (func->IsInstance<FunctionNode>()) {
        auto updated_func = Downcast<Function>(this->VisitExpr(func));
        builder_->UpdateFunction(gv, Downcast<BaseFunc>(updated_func));
      }
    }

    IRModule output = builder_->GetContextIRModule();
    if (generated_tir_with_target_attr_) {
      // It is possible that every call to a legalized PrimFunc
      // contains VDevice annotations.  In that case, the PrimFunc
      // without a target annotation no longer has any callers, and
      // should be removed.
      output = relax::transform::DeadCodeElimination()(output);

      // Avoid accidental sharing of TIR variables in the legalized
      // PrimFuncs, when kernels for multiple devices are generated
      // from the same PrimFunc.
      output = tir::transform::ConvertSSA()(output);
    }

    return output;
  }

 private:
  using ExprMutator::VisitExpr_;

  bool WrapPureCondition(const Op& op, const Expr& legalized) {
    static const auto& purity_map = Op::GetAttrMap<Bool>("FPurity");

    const CallNode* call = legalized.as<CallNode>();

    if (!call) {
      // Unlikely for this condition to be met, but it is possible.
      // For example, an operation could produce a Tuple output, and
      // be legalized into separate calls for each item in the Tuple.
      return false;
    }

    bool pure_original_op = purity_map.get(op, Bool(false))->value;
    bool pure_legalized_op = [&]() -> bool {
      if (auto legalized_op = call->op.as<Op>()) {
        return purity_map.get(legalized_op.value(), Bool(false))->value;
      } else if (auto func_sinfo = call->op->struct_info_.as<FuncStructInfoNode>()) {
        return func_sinfo->purity;
      } else {
        return false;
      }
    }();

    // If the original op was pure, but the legalized op was not,
    // the legalized op may occur in a context that requires pure
    // functions, such as a `relax::DataflowBlock`.  In this case,
    // we should wrap the legalized operation to indicate that it is
    // still pure.
    return pure_original_op && !pure_legalized_op;
  }

  Call WrapPureCall(const Call& ret) {
    static const Op& call_pure_packed_op = Op::Get("relax.call_pure_packed");
    Array<Expr> ret_args = {ret->op};
    for (auto arg : ret->args) {
      ret_args.push_back(arg);
    }
    return Call(call_pure_packed_op, ret_args, ret->attrs, ret->sinfo_args);
  }

  Optional<Target> GetTarget(const Array<StructInfo>& sinfos) {
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
    return NullOpt;
  }

  Expr BindTarget(Expr expr) {
    if (!expr->IsInstance<CallNode>()) {
      // FLegalize returned something other than a relax::Call.  This
      // post-processing only handles cases where legalization
      // produces a lowered call node.  In principle, this
      // post-processing isn't necessary, and FLegalize should already
      // have generated vdevice-aware kernels, so hopefully the
      // FLegalize implementation did so.
      return expr;
    }

    auto call = Downcast<Call>(expr);

    auto vdevice_target = GetTarget(call->sinfo_args);
    if (!vdevice_target.defined()) {
      // No vdevice annotation is present, so we don't need to apply
      // any updates.
      return expr;
    }

    if (call->args.empty()) {
      return expr;
    }

    auto gvar = call->args[0].as<GlobalVar>();
    if (!gvar.defined()) {
      // This is not a call into a legalized function within the
      // current IRModule, so no post-processing is required.
      return expr;
    }

    auto base_func = builder_->GetContextIRModule()->Lookup(gvar.value());
    auto opt_prim_func = base_func.as<tir::PrimFunc>();
    if (!opt_prim_func) {
      // The call is to something other than a PrimFunc.  It may be
      // another Relax function, in which case the legalization of its
      // body will handle any additional target annotations.
      return expr;
    }
    auto prim_func = opt_prim_func.value();

    auto func_target = prim_func->GetAttr<Target>(tvm::attr::kTarget);
    if (func_target && func_target.value()->kind == vdevice_target.value()->kind) {
      // The function already has compatible annotations for the
      // target, so no modifications are required.
      return expr;
    }

    // The FLegalize function generated a PrimFunc, but that PrimFunc
    // doesn't have annotations compatible with the vdevice required
    // by the Relax StructInfo.  Update the call to instead call a
    // `PrimFunc` with the appropriate target annotation.  In the
    // future, this may be treated as a bug in the FLegalize
    // implementation, rather than expected output from it.
    auto new_prim_func = WithAttr(prim_func, tvm::attr::kTarget, vdevice_target.value());
    auto new_gvar_name = [&]() -> std::string {
      std::stringstream ss;
      ss << gvar.value()->name_hint;
      ss << "_";
      ss << vdevice_target.value()->kind->name;
      return ss.str();
    }();
    auto new_gvar = builder_->AddFunction(new_prim_func, new_gvar_name);
    generated_tir_with_target_attr_ = true;

    call.CopyOnWrite()->args.Set(0, new_gvar);
    return call;
  }

  Expr VisitExpr_(const CallNode* call) final {
    Call visited_call = Downcast<Call>(this->VisitExprPostOrder_(call));
    static const auto& legalize_map = Op::GetAttrMap<FLegalize>("FLegalize");
    static const auto& call_packed_map = Op::GetAttrMap<FCallPacked>("FCallPacked");
    static const auto& requires_arg_shapes_map = Op::GetAttrMap<Bool>("RequiresArgumentShapes");
    static const Op& call_pure_packed_op = Op::Get("relax.call_pure_packed");
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    static const Op& call_dps_packed_op = Op::Get("relax.call_dps_packed");
    auto* op_node = visited_call->op.as<OpNode>();

    // Not an OpNode
    if (op_node == nullptr) {
      return visited_call;
    }
    auto op = GetRef<Op>(op_node);

    bool shapes_are_known_if_required = [&]() -> bool {
      bool requires_arg_shapes = requires_arg_shapes_map.get(op, Bool(true))->value;
      if (!requires_arg_shapes) {
        // This operator does not require its arguments to have a
        // known shape/dtype.  For example, the "relax.tensor_ndim"
        // operator can output the dimensionality of a tensor at
        // runtime, and does not require the dimensionality to be
        // known at compile-time.
        return true;
      }

      bool arg_shapes_defined =
          std::all_of(visited_call->args.begin(), visited_call->args.end(),
                      [](Expr arg) { return KnowAllShapeValues(GetStructInfo(arg)); });
      if (!arg_shapes_defined) {
        // This operator cannot be legalized, because legalization
        // requires the argument shapes to be known.
        //
        // TODO(Lunderberg):
        //
        //     Improve this fallback case, as failure to legalize can
        //     produce unexpected errors during CodeGenVM.  This could
        //     be done by having `R.Tensor(ndim=2)` be syntactic sugar
        //     for `R.Tensor(shape=[m, n])`, where `m` and `n` are new
        //     shape variables.  This would allow legalization into
        //     dynamic TIR PrimFuncs.
        //
        //     This fallback would only be applicable for cases where
        //     both the dtype and the dimensionality are known.  While
        //     Relax can express a tensor with unknown dtype and
        //     dimensionality as `TensorStructInfo(DataType::Void(),
        //     kUnknownNDim)`, TIR cannot express unknown dtype or
        //     unknown dimensionality.
        return false;
      }

      std::string op_name(op->name);
      bool is_data_dependent_op = (op_name.find("dynamic") != std::string::npos);
      bool ret_shape_defined = KnowAllShapeValues(GetStructInfo(visited_call));
      if (!is_data_dependent_op && !ret_shape_defined) {
        // This operator cannot be legalized, because legalization by
        // default requires the output shape.  The exception is
        // data-dependent operators (e.g. `R.dynamic_strided_slice`),
        // where the shape of the output depends on the runtime values
        // stored in a tensor.
        //
        // For data-dependent ops, the output shape will be identified
        // at runtime.  The Legalizer will insert their shape
        // functions, which are manually registered for each
        // data-dependent op, and match cast to define symbolic output
        // shapes.  These symbolic output shapes at compile time can
        // be by later operations to refer to the runtime shape.
        //
        // TODO(Lunderberg): Make a new operator attribute
        // `.set_attr<Bool>("DataDependent")`, rather than relying on
        // the name of the operator.
        return false;
      }

      // All checks pass, this operator can be legalized.
      return true;
    }();

    FLegalize legalization_func;

    if (auto opt_custom_legalize = cmap_.Get(op->name);
        opt_custom_legalize && shapes_are_known_if_required) {
      // First choice, use a custom legalization function
      legalization_func = opt_custom_legalize.value();
    } else if (legalize_map.count(op) && shapes_are_known_if_required) {
      // Second choice, use a default legalization
      legalization_func = legalize_map[op];
    } else if (call_packed_map.count(op)) {
      // Third choice, use an explicit FCallPacked replacement.  This does not require the shape
      String packed_func_name = call_packed_map[op];
      legalization_func = [packed_func_name](const BlockBuilder& bb, const Call& call) -> Expr {
        return Call(ExternFunc(packed_func_name), call->args, Attrs(), {GetStructInfo(call)});
      };
    } else {
      // No legalization.
      if (enable_warning_ && op != call_tir_op && op != call_dps_packed_op &&
          op != call_pure_packed_op) {
        if (shapes_are_known_if_required) {
          LOG(WARNING) << "No legalization func for " << op->name << " is found.";
        } else {
          LOG(WARNING) << "Cannot legalize " << visited_call
                       << ", missing known shapes for arguments and return value";
        }
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

    // Append the target attribute to any PrimFunc generated in
    // legalization.
    legalized = BindTarget(legalized);

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
  /*! \brief If VDevice annotations produced at least one PrimFunc with a Target attr*/
  bool generated_tir_with_target_attr_{false};
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

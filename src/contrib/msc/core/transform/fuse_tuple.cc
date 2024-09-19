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
 * \file src/contrib/msc/core/transform/fuse_tuple.cc
 * \brief Pass for fuse ShapeExpr.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../../../../relax/transform/utils.h"
#include "../utils.h"

namespace tvm {
namespace relax {

using namespace tvm::contrib::msc;

/*!
 * \brief Fuse Tuple and TupleGetItem to BYOC
 */
class TupleFuser : public ExprMutator {
 public:
  explicit TupleFuser(IRModule ctx_module, const String& target, const String& entry_name)
      : ExprMutator(ctx_module) {
    mod_ = ctx_module;
    target_ = target + ".";
    entry_name_ = entry_name;
  }

  IRModule Fuse() {
    GlobalVar main_var;
    for (const auto& [gv, func] : mod_->functions) {
      if (gv->name_hint == entry_name_) {
        main_var = gv;
      } else {
        const auto& name_opt = func->GetAttr<runtime::String>(attr::kComposite);
        if (name_opt.defined() && StringUtils::StartsWith(name_opt.value(), target_)) {
          target_funcs_.Set(gv, Downcast<Function>(func));
        }
      }
    }
    // update main
    ICHECK(main_var.defined()) << "Can not find entry func " << entry_name_;
    const auto& new_func = Downcast<Function>(VisitExpr(mod_->Lookup(entry_name_)));
    builder_->UpdateFunction(main_var, new_func);
    return builder_->GetContextIRModule();
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) final {
    bool has_tuple_arg = false;
    if (target_funcs_.count(val->op)) {
      Array<Expr> new_args;
      for (size_t i = 0; i < val->args.size(); i++) {
        const auto& arg = val->args[i];
        if (arg->IsInstance<TupleNode>()) {
          String tuple_name;
          const auto& name_opt =
              target_funcs_[val->op]->GetAttr<runtime::String>(msc_attr::kUnique);
          if (name_opt.defined()) {
            if (val->args.size() == 1) {
              tuple_name = name_opt.value() + "_input";
            } else {
              tuple_name = name_opt.value() + "_inputs." + std::to_string(i);
            }
          }
          const auto& func_call = AddFunc(arg, tuple_name);
          const auto& tuple_out = builder_->Emit(func_call);
          ICHECK(target_funcs_.count(func_call->op))
              << "Can not find target func " << func_call->op;
          target_funcs_.Set(tuple_out, target_funcs_[func_call->op]);
          has_tuple_arg = true;
          new_args.push_back(tuple_out);
        } else {
          new_args.push_back(arg);
        }
        if (has_tuple_arg) {
          const auto& new_call = Call(val->op, new_args, val->attrs, val->sinfo_args, val->span);
          ReEmitBinding(binding, builder_->Normalize(new_call));
        }
      }
      target_funcs_.Set(binding->var, target_funcs_[val->op]);
    }
    if (!has_tuple_arg) {
      ExprMutator::VisitBinding_(binding, val);
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) final {
    bool on_target = true;
    for (const auto& f : val->fields) {
      if (!target_funcs_.count(f)) {
        on_target = false;
        break;
      }
    }
    if (on_target) {
      ReEmitFunc(binding, GetRef<Tuple>(val));
    } else {
      ExprMutator::VisitBinding_(binding, val);
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) final {
    if (target_funcs_.count(val->tuple)) {
      ReEmitFunc(binding, GetRef<TupleGetItem>(val));
    } else {
      ExprMutator::VisitBinding_(binding, val);
    }
  }

 private:
  Call AddFunc(const Expr& expr, const String tuple_name = "") {
    builder_->BeginDataflowBlock();
    Array<Expr> inputs;
    if (const auto* v_node = expr.as<TupleNode>()) {
      inputs = v_node->fields;
    } else if (const auto* g_node = expr.as<TupleGetItemNode>()) {
      inputs = {g_node->tuple};
    } else {
      LOG_FATAL << "Unexpceted expr " << expr;
    }
    Array<Expr> func_inputs;
    Array<Expr> call_inputs;
    Array<Var> params;
    Map<Expr, Var> added_params;
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i]->IsInstance<ConstantNode>()) {
        func_inputs.push_back(inputs[i]);
        continue;
      }
      if (!added_params.count(inputs[i])) {
        const auto& name = String("param_" + std::to_string(i));
        const auto& var = Var(std::move(name), GetStructInfo(inputs[i]));
        added_params.Set(inputs[i], var);
      }
      call_inputs.push_back(inputs[i]);
      func_inputs.push_back(added_params[inputs[i]]);
      params.push_back(added_params[inputs[i]]);
    }

    Expr out_expr;
    String func_name;
    Span expr_span = expr->span;
    if (!expr_span.defined()) {
      ICHECK(tuple_name.size() > 0) << "Missing tuple for " << expr;
      expr_span = SpanUtils::CreateWithAttr(msc_attr::kName, tuple_name);
    }
    if (expr->IsInstance<TupleNode>()) {
      out_expr = Tuple(func_inputs, expr_span);
      func_name = "tuple";
    } else if (const auto* g_node = expr.as<TupleGetItemNode>()) {
      out_expr = TupleGetItem(func_inputs[0], g_node->index, expr_span);
      func_name = "get_item";
    } else {
      LOG_FATAL << "Unexpceted expr " << expr;
    }

    const auto& output = builder_->EmitOutput(out_expr);
    BindingBlock new_block = builder_->EndBlock();
    Expr body = builder_->Normalize(output);
    body = builder_->Normalize(SeqExpr({new_block}, body));

    Map<String, ObjectRef> func_attrs;
    func_attrs.Set(attr::kPrimitive, Integer(1));
    func_attrs.Set(attr::kComposite, target_ + func_name);
    func_attrs.Set(msc_attr::kUnique, SpanUtils::GetAttr(expr_span, msc_attr::kName));

    Function function = Function(/*params=*/params,            //
                                 /*body=*/body,                //
                                 /*ret_struct_info=*/NullOpt,  //
                                 /*is_pure=*/true,             //
                                 /*attrs=*/DictAttrs(func_attrs));
    Array<PrimExpr> free_vars =
        FreeSymbolicVars(function).Map([](const tir::Var& var) -> PrimExpr { return var; });
    if (!free_vars.empty()) {
      params.push_back(Var("tir_vars", ShapeStructInfo(free_vars)));
      function = Function(/*params=*/params,            //
                          /*body=*/body,                //
                          /*ret_struct_info=*/NullOpt,  //
                          /*is_pure=*/true,             //
                          /*attrs=*/DictAttrs(func_attrs));
    }
    function = SymbolicVarRenewMutator::Renew(function);
    GlobalVar gv = builder_->AddFunction(function, "fused_" + func_name);
    target_funcs_.Set(gv, function);
    return Call(gv, call_inputs);
  }

  void ReEmitFunc(const VarBindingNode* binding, const Expr& expr) {
    const auto& func_call = AddFunc(expr);
    ReEmitBinding(binding, builder_->Normalize(func_call));
    ICHECK(target_funcs_.count(func_call->op)) << "Can not find target func " << func_call->op;
    target_funcs_.Set(binding->var, target_funcs_[func_call->op]);
  }

  IRModule mod_;
  String target_;
  String entry_name_;
  Map<Expr, Function> target_funcs_;
};

IRModule FuseTuple(IRModule mod, const String& target, const String& entry_name) {
  return TupleFuser(mod, target, entry_name).Fuse();
}

namespace transform {

Pass FuseTuple(const String& target, const String& entry_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::FuseTuple(m, target, entry_name); };
  return CreateModulePass(pass_func, 0, "FuseTuple", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FuseTuple").set_body_typed(FuseTuple);

}  // namespace transform
}  // namespace relax
}  // namespace tvm

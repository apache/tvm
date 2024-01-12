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
 * \file tvm/relax/transform/lambda_lift.cc
 * \brief Lift local functions into global functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

namespace tvm {
namespace relax {

namespace {

/* \brief Collect names of functions to be lifted out */
class LambdaNameCollector : ExprVisitor {
 public:
  static std::unordered_map<const FunctionNode*, String> Collect(const IRModule& mod) {
    LambdaNameCollector visitor;

    for (const auto& [gvar, base_func] : mod->functions) {
      visitor.previous_global_vars_.insert(gvar->name_hint);
    }

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        visitor.name_stack_.push_back(gvar->name_hint);
        visitor(func.value());
        visitor.name_stack_.pop_back();
      }
    }

    return visitor.Finalize();
  }

 private:
  void VisitBinding_(const VarBindingNode* binding, const FunctionNode* func) override {
    if (auto opt = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      String public_name = opt.value();

      // If a kGlobalSymbol exists, we must use the name exactly as it
      // appears, with no modifications.  Because these errors would
      // be raised from deep within an optimization pipeline, but
      // depends on small annotation changes from a user's initial
      // model definition, they are intentionally verbose to
      // (hopefully) provide sufficient context to a user encountering
      // the error.
      CHECK(!previous_global_vars_.count(public_name))
          << "Function " << name_stack_.front() << " contains a lambda with kGlobalSymbol (\""
          << tvm::attr::kGlobalSymbol << "\" attribute of \"" << public_name << "\".  "
          << "However, the module already contains a GlobalVar with this name.  "
          << "If present, the kGlobalSymbol attribute must match the name of the GlobalVar, "
          << "and GlobalVar names must be unique across an IRModule.  "
          << "Lifting the " << public_name << " function out of " << name_stack_.front()
          << " would require violating one of these two conditions.";

      auto it = new_public_names_.find(public_name);
      CHECK(it == new_public_names_.end())
          << "Function " << name_stack_.front() << " contains a lambda with kGlobalSymbol (\""
          << tvm::attr::kGlobalSymbol << "\" attribute of \"" << public_name << "\".  "
          << "However, the function " << it->second.front()
          << " also contains a lambda with the same value for kGlobalSymbol.  "
          << "If present, the kGlobalSymbol attribute must match the name of the GlobalVar, "
          << "and GlobalVar names must be unique across an IRModule.  "
          << "Lifting the " << public_name << " function out of both " << name_stack_.front()
          << " and " << it->second.front()
          << " would require violating one of these two conditions.";

      new_public_names_.insert({public_name, name_stack_});
      lifted_with_global_symbol_.insert({func, public_name});
    }

    name_stack_.push_back(binding->var->name_hint());
    lambda_location_.insert({func, name_stack_});
    ExprVisitor::VisitBinding_(binding, func);
    name_stack_.pop_back();
  }

  // De-duplication of collected names
  std::unordered_map<const FunctionNode*, String> Finalize() const {
    // The functions which still must be assigned a name
    std::unordered_map<const FunctionNode*, Array<String>> remaining_to_name = lambda_location_;

    // Collecting the functions that now have a name.
    std::unordered_map<const FunctionNode*, String> lifted_names;

    // A lookup for names that are unavailable for use.
    std::unordered_set<String> unavailable_names = previous_global_vars_;

    // A helper function to generate de-duplicated names.  The
    // `proposed_name_generation_func` should be a function with
    // signature:
    //
    //     Optional<String> func(const FunctionNode*, const Array<String>&)
    //
    // The first argument will be the lambda function being lifted.
    // The second argument will be the nested location where that
    // lambda function was found.  The function should return the
    // proposed name for the lifted lambda function.  The proposed
    // name will be accepted if it does not conflict with any previous
    // names, and is unique for all lambda functions being lifted.
    //
    // This helper function is used to apply several different schemes
    // to generate the name of the lifted lambda function.  The
    // overall goal is to provide names that are unique (required by
    // IRModule), deterministic (required for unit testing), and
    // human-readable.
    auto attempt_name_generation = [&](const auto& proposed_name_generation_func) {
      if (remaining_to_name.empty()) {
        return;
      }

      std::unordered_map<String, const FunctionNode*> new_names;
      for (const auto& [func, location] : remaining_to_name) {
        if (Optional<String> opt_proposed_name = proposed_name_generation_func(func, location)) {
          auto proposed_name = opt_proposed_name.value();

          if (unavailable_names.count(proposed_name)) {
            // The name is already used, either from a GlobalVar, or
            // from a previous round of attempted names.
          } else if (auto it = new_names.find(proposed_name); it != new_names.end()) {
            // The name is not unique within the current attempt.  Mark
            // the function as nullptr to previous any use of this name
            it->second = nullptr;
          } else {
            // The name is unique so far.  Track it for use.
            new_names.insert({proposed_name, func});
          }
        }
      }

      for (const auto& [name, func] : new_names) {
        if (func) {
          lifted_names.insert({func, name});
          remaining_to_name.erase(func);
        }
      }
    };

    // 1. Start with any publicly explosed names from kGlobalSymbol
    attempt_name_generation([&](const FunctionNode* func, const auto&) -> Optional<String> {
      if (auto it = lifted_with_global_symbol_.find(func); it != lifted_with_global_symbol_.end()) {
        return it->second;
      } else {
        return NullOpt;
      }
    });

    // 2. Try concatenating the name of the relax variable with the
    // name of the function that contains it.
    attempt_name_generation([&](const FunctionNode*, const auto& location) -> String {
      std::stringstream stream;
      stream << location.front() << "_" << location.back();
      return stream.str();
    });

    // 3. Try concatenating the entire path together.  Don't include
    // paths of length 2, as they would already be attempted earlier.
    attempt_name_generation([&](const FunctionNode*, const auto& location) -> Optional<String> {
      if (location.size() == 2) return NullOpt;

      std::stringstream stream;
      bool is_first = true;
      for (const auto& loc : location) {
        if (is_first) {
          is_first = false;
        } else {
          stream << "_";
        }
        stream << loc;
      }
      return String(stream.str());
    });

    // 4. Fallback.  Count the number of times a relax variable with
    // that name was used.
    std::unordered_map<String, int> usage_count;
    attempt_name_generation([&](const FunctionNode*, const auto& location) -> String {
      std::stringstream stream;
      stream << location.front() << "_" << location.back();
      int usage = usage_count[stream.str()]++;
      stream << "_" << usage;

      return stream.str();
    });

    ICHECK(remaining_to_name.empty())
        << "Fallback failed to make unique names for all lifted lambda functions";

    return lifted_names;
  }

  Array<String> name_stack_;
  std::unordered_set<String> previous_global_vars_;
  std::unordered_map<String, Array<String>> new_public_names_;
  std::unordered_map<const FunctionNode*, String> lifted_with_global_symbol_;
  std::unordered_map<const FunctionNode*, Array<String>> lambda_location_;
};

}  // namespace

/* The goal of this class is to lift out any nested functions into top-level
 * functions.
 *
 * We will lift a function out into a global which takes the set of the free
 * vars and then return the new created function.
 */
class LambdaLifter : public ExprMutator {
 public:
  explicit LambdaLifter(const IRModule& module)
      : ExprMutator(module), mod_(module), lifted_names_(LambdaNameCollector::Collect(module)) {}

  using ExprMutator::VisitExpr_;

  void VisitBinding_(const VarBindingNode* binding) final {
    bool is_lambda = binding->value->IsInstance<FunctionNode>();
    if (is_lambda) {
      recur_vars_.push_back(binding->var);
    }

    Expr new_value = this->VisitExpr(binding->value);

    if (new_value->struct_info_.defined() &&
        !new_value->struct_info_.same_as(binding->var->struct_info_)) {
      binding->var->struct_info_ = GetStructInfo(new_value);
      binding->var->checked_type_ = new_value->checked_type_;
    }
    if (new_value.same_as(binding->value)) {
      builder_->EmitNormalized(GetRef<VarBinding>(binding));
    } else {
      builder_->EmitNormalized(VarBinding(binding->var, new_value));
    }
    if (is_lambda) {
      recur_vars_.pop_back();
    }
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    if (const auto* var_node = call_node->op.as<VarNode>()) {
      auto var = GetRef<Var>(var_node);
      bool has_closure = HasClosure(var);
      auto val = builder_->LookupBinding(var);
      if (const auto* fsinfo_node = GetStructInfo(var).as<FuncStructInfoNode>()) {
        auto fsinfo = GetRef<FuncStructInfo>(fsinfo_node);
        if (!GetStructInfo(call).same_as(fsinfo)) {
          call->struct_info_ = fsinfo->ret;
          call->checked_type_ = GetStaticType(fsinfo->ret);
        }
      }
      // Call "relax.invoke_closure" to invoke closure
      Var clo_arg = var;
      if (has_closure && val->IsInstance<CallNode>()) {
        if (this->var_remap_.find(var->vid) != this->var_remap_.end()) {
          clo_arg = this->var_remap_.at(var->vid);
        }

        // if the original op was pure, we should use invoke_pure_closure
        Call orig_call = Downcast<Call>(val);
        bool purity;
        if (orig_call->op.as<OpNode>()) {
          auto orig_op = Downcast<Op>(orig_call->op);
          static const auto& purity_map = Op::GetAttrMap<Bool>("FPurity");
          purity = purity_map.count(orig_op) && purity_map[orig_op]->value;
        } else {
          purity = GetStructInfoAs<FuncStructInfoNode>(orig_call->op)->purity;
        }

        return Call(purity ? invoke_pure_closure_op_ : invoke_closure_op_,
                    {clo_arg, Tuple(call_node->args)}, {},
                    {GetStructInfo(GetRef<Expr>(call_node))});
      }
      auto it = lambda_map_.find(var);
      if (it != lambda_map_.end()) {
        // flatten nested call, e.g. call(y)(x) -> call(x, y))
        Array<relay::Expr> new_args;
        Array<StructInfo> params;
        for (const auto arg : call->args) {
          new_args.push_back(arg);
          params.push_back(StructInfoFromType(arg->checked_type()));
        }
        if (const auto* nest_call = it->second.as<CallNode>()) {
          // Update the StructInfo accordingly
          for (const auto arg : nest_call->args) {
            new_args.push_back(arg);
            params.push_back(StructInfoFromType(arg->checked_type()));
          }
          StructInfo new_func_sinfo;
          if (const auto* fsinfo = GetStructInfo(nest_call->op).as<FuncStructInfoNode>()) {
            auto func_sinfo = GetRef<FuncStructInfo>(fsinfo);
            new_func_sinfo = FuncStructInfo(params, func_sinfo->ret);
          }
          nest_call->op->struct_info_ = new_func_sinfo;
          nest_call->op->checked_type_ = GetStaticType(new_func_sinfo);
          return Call(nest_call->op, new_args, call_node->attrs, call_node->sinfo_args);
        }
        return Call(it->second, call->args, call_node->attrs, call_node->sinfo_args);
      }
    }
    return std::move(call);
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);

    String lift_func_name = [&]() {
      auto it = lifted_names_.find(func_node);
      ICHECK(it != lifted_names_.end())
          << "InternalError: "
          << "Found lambda function during mutation step, "
          << "but it wasn't found during the earlier name-generation step.";
      return it->second;
    }();

    auto global = GlobalVar(lift_func_name);
    Array<Var> free_vars = FreeVars(func);
    Array<Var> captured_vars;

    Array<Var> typed_captured_vars;
    bool recursive = false;
    for (const auto& var : free_vars) {
      if (!recur_vars_.empty() && var == recur_vars_.back()) {
        recursive = true;
      } else {
        captured_vars.push_back(var);
      }
    }

    Map<Var, Expr> rebinding_map;
    for (auto free_var : captured_vars) {
      Var var = Var(free_var->name_hint(), GetStructInfo(free_var), free_var->span);
      typed_captured_vars.push_back(var);
      rebinding_map.Set(free_var, var);
    }

    // recursive call
    if (recursive) {
      if (!captured_vars.empty()) {
        Array<Expr> fvs;
        for (auto fv : captured_vars) {
          fvs.push_back(fv);
        }
        // it is required by block_blocker, will be updated later
        UpdateStructInfo(global, GetStructInfo(recur_vars_.back()));
        lambda_map_.emplace(recur_vars_.back(), Call(global, fvs));
      } else {
        if (recur_vars_.size() > 0) {
          lambda_map_.emplace(recur_vars_.back(), global);
        }
      }
    }

    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    for (Var param : func_node->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      all_params_unchanged &= param.same_as(new_param);
    }

    Expr body = this->VisitWithNewScope(func_node->body);
    Expr visited_func;

    if (all_params_unchanged && body.same_as(func_node->body)) {
      visited_func = GetRef<Expr>(func_node);
    } else if (const auto& body_sinfo = MatchStructInfo<ObjectStructInfo>(body)) {
      visited_func =
          Function(params, body, body_sinfo.value(), func_node->is_pure, func_node->attrs);
    } else {
      visited_func =
          Function(params, body, func_node->ret_struct_info, func_node->is_pure, func_node->attrs);
    }
    auto new_func = Downcast<Function>(visited_func);

    Function lifted_func;
    bool is_closure = IsClosure(captured_vars);
    if (!is_closure) {
      lifted_func = Function(
          /*params=*/new_func->params,
          /*body=*/new_func->body,
          /*ret_struct_info=*/new_func->ret_struct_info,
          /*is_pure=*/new_func->is_pure,
          /*attrs=*/new_func->attrs,
          /*span=*/new_func->span);
    } else {
      // Flatten the Closure
      std::vector<Var> closure_params;
      closure_params.reserve(func->params.size() + typed_captured_vars.size());
      for (size_t i = 0; i < func->params.size(); ++i) {
        closure_params.emplace_back(func->params[i]);
      }
      for (size_t i = 0; i < typed_captured_vars.size(); ++i) {
        closure_params.emplace_back(typed_captured_vars[i]);
      }

      lifted_func = Function(/*params=*/closure_params,
                             /*body=*/Bind(new_func->body, rebinding_map),
                             /*ret_struct_info=*/new_func->ret_struct_info,
                             /*is_pure=*/new_func->is_pure,
                             /*attrs=*/new_func->attrs,
                             /*span=*/func->span);

      for (Var param : closure_params) {
        CHECK(param->checked_type_.defined())
            << "relax.Function requires params to contain checked_type_";
      }
    }

    ICHECK(lifted_func.defined());

    // Add the lifted function to the module.
    global->struct_info_ = GetStructInfo(lifted_func);
    global->checked_type_ = lifted_func->checked_type_;
    builder_->UpdateFunction(global, lifted_func);

    if (!is_closure) {
      return std::move(global);
    } else {
      // If we need to allocate a closure,
      // we pass the variables in its environment here.
      Array<Expr> fvs;
      for (auto fv : captured_vars) {
        fvs.push_back(fv);
      }
      // Call make_closure intrinsic
      return Call(make_closure_op_, {global, Tuple(fvs)}, {}, {});
    }
  }

  bool HasClosure(const Var& var) {
    auto val = builder_->LookupBinding(var);
    if (const auto* value = val.as<GlobalVarNode>()) {
      IRModule ctx_mod = builder_->GetContextIRModule();
      ICHECK(ctx_mod->functions.size() > 0);
      BaseFunc func = ctx_mod->Lookup(GetRef<GlobalVar>(value));
      if (const auto* func_node = func.as<FunctionNode>()) {
        if (const auto* call_node = func_node->body.as<CallNode>()) {
          if (call_node->op == make_closure_op_) {
            return true;
          }
        } else if (const auto* seq_expr_node = func_node->body.as<SeqExprNode>()) {
          // the return var points to a make_closure intrinsic
          if (const auto* var = seq_expr_node->body.as<VarNode>()) {
            return HasClosure(GetRef<Var>(var));
          }
        }
      }
    } else if (const auto* func_node = val.as<FunctionNode>()) {
      if (const auto* call_node = func_node->body.as<CallNode>()) {
        if (call_node->op == make_closure_op_) {
          return true;
        }
      }
    } else if (const auto* call_node = val.as<relax::CallNode>()) {
      // recursive call
      auto op = call_node->op;
      if (make_closure_op_ == op) {
        return true;
      }
      if (const auto* lv = op.as<VarNode>()) {
        return HasClosure(GetRef<Var>(lv));
      }
    }
    return false;
  }

  bool IsClosure(const Array<Var>& captured_vars) { return captured_vars.size() > 0; }

  IRModule Lift() {
    auto glob_funcs = mod_->functions;
    for (auto pair : glob_funcs) {
      if (auto* n = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(n);
        func = Function(func->params, VisitExpr(func->body), func->ret_struct_info, func->is_pure,
                        func->attrs);
        builder_->UpdateFunction(pair.first, func);
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> lambda_map_;
  Array<Var> recur_vars_;
  IRModule mod_;

  std::unordered_map<const FunctionNode*, String> lifted_names_;

  /*! \brief Cache ops that would be used later to reduce lookup overhead. */
  const Op& make_closure_op_ = Op::Get("relax.make_closure");
  const Op& invoke_closure_op_ = Op::Get("relax.invoke_closure");
  const Op& invoke_pure_closure_op_ = Op::Get("relax.invoke_pure_closure");
};

namespace transform {

Pass LambdaLift() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::LambdaLifter(m).Lift(); };
  return CreateModulePass(pass_func, 1, "LambdaLift", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LambdaLift").set_body_typed(LambdaLift);

}  // namespace transform
}  // namespace relax
}  // namespace tvm

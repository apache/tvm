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

  void VisitBinding_(const VarBindingNode* binding, const FunctionNode* func_node) final {
    auto cache = current_lambda_var_;
    current_lambda_var_ = binding->var;

    auto new_value = VisitExpr(binding->value);
    if (!rebind_map_.count(binding->var)) {
      ReEmitBinding(binding, new_value);
    }

    current_lambda_var_ = cache;
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    if (!current_lambda_var_) {
      // Early bail-out for top-level functions
      return ExprMutator::VisitExpr_(func_node);
    }

    auto func = GetRef<Function>(func_node);

    String lift_func_name = [&]() {
      auto it = lifted_names_.find(func_node);
      ICHECK(it != lifted_names_.end())
          << "InternalError: "
          << "Found lambda function during mutation step, "
          << "but it wasn't found during the earlier name-generation step.";
      return it->second;
    }();

    Array<Var> captured_vars;
    bool is_recursive = false;
    bool is_closure = false;
    for (const auto& var : FreeVars(func)) {
      if (var.same_as(current_lambda_var_)) {
        is_recursive = true;
      } else {
        is_closure = true;
        captured_vars.push_back(var);
      }
    }

    Array<Var> typed_captured_vars;
    Map<Var, Expr> rebinding_map;
    for (auto free_var : captured_vars) {
      Var var = Var(free_var->name_hint(), GetStructInfo(free_var), free_var->span);
      typed_captured_vars.push_back(var);
      rebinding_map.Set(free_var, var);
    }

    tvm::Array<Var> lifted_func_params =
        func_node->params.Map([this](Var var) { return VisitVarDef(var); });
    for (const auto& var : typed_captured_vars) {
      lifted_func_params.push_back(var);
    }

    auto gvar_lifted_func = GlobalVar(lift_func_name);
    {
      auto func_sinfo = Downcast<FuncStructInfo>(func_node->struct_info_);
      if (is_closure) {
        func_sinfo = FuncStructInfo(lifted_func_params.Map(GetStructInfo), func_sinfo->ret,
                                    func_sinfo->purity);
      }
      UpdateStructInfo(gvar_lifted_func, func_sinfo);
    }

    Expr body = func_node->body;

    // Defining the rewrite rule prior to visiting the body, so that
    // recursive closures can be updated.
    if (is_recursive && is_closure) {
      nested_closure_map_.emplace(
          current_lambda_var_.value(),
          Call(gvar_lifted_func, captured_vars.Map([](Var var) -> Expr { return var; })));
    }

    if (!is_closure) {
      rebind_map_.emplace(current_lambda_var_.value(), gvar_lifted_func);
    }

    body = this->VisitWithNewScope(body, lifted_func_params);
    StructInfo ret_struct_info = GetStructInfo(body);
    body = Bind(body, rebinding_map);

    Function lifted_func;
    if (lifted_func_params.same_as(func_node->params) && body.same_as(func_node->body) &&
        ret_struct_info.same_as(func_node->ret_struct_info)) {
      lifted_func = GetRef<Function>(func_node);
    } else {
      lifted_func =
          Function(lifted_func_params, body, ret_struct_info, func_node->is_pure, func_node->attrs);
    }

    for (Var param : lifted_func->params) {
      CHECK(param->checked_type_.defined())
          << "relax.Function requires all parameters to contain checked_type_.  "
          << "However, parameter " << param << " with struct info " << param->struct_info_
          << " has no checked type";
    }

    ICHECK(lifted_func.defined());

    if (is_closure || IsClosure(lifted_func)) {
      closures_.insert(gvar_lifted_func);
    }

    // Add the lifted function to the module.
    lifted_func = CopyWithNewVars(lifted_func);
    gvar_lifted_func->struct_info_ = GetStructInfo(lifted_func);
    gvar_lifted_func->checked_type_ = lifted_func->checked_type_;

    builder_->UpdateFunction(gvar_lifted_func, lifted_func);

    Expr callable_value = gvar_lifted_func;
    if (is_closure) {
      // If we need to allocate a closure,
      // we pass the variables in its environment here.
      Tuple arg_tuple(captured_vars.Map([](Var var) -> Expr { return var; }));
      // Call make_closure intrinsic
      callable_value = Call(make_closure_op_, {gvar_lifted_func, arg_tuple}, {}, {});
    }

    return callable_value;
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = GetRef<Call>(call_node);

    auto orig_sinfo = Downcast<StructInfo>(call->struct_info_);

    if (auto opt_var = call->op.as<Var>()) {
      auto var = opt_var.value();

      // Call "relax.invoke_closure" to invoke closure

      if (IsClosure(var) && builder_->LookupBinding(var).as<CallNode>()) {
        // if the original op was pure, we should use invoke_pure_closure
        Call orig_call = Downcast<Call>(builder_->LookupBinding(var));
        bool is_pure = [&]() -> bool {
          if (auto op = orig_call->op.as<Op>()) {
            static const auto& purity_map = Op::GetAttrMap<Bool>("FPurity");
            return purity_map.get(op.value(), Bool(false))->value;
          } else if (const auto* func_sinfo =
                         orig_call->op->struct_info_.as<FuncStructInfoNode>()) {
            return func_sinfo->purity;
          } else {
            LOG(FATAL) << "Could not determine purity of call to " << orig_call->op
                       << ", as it is neither a tvm::Op (type = \"" << orig_call->op->GetTypeKey()
                       << "\"), "
                       << "nor is is annotated with FuncStructInfo (sinfo = "
                       << orig_call->op->struct_info_ << ")";
          }
        }();

        auto prev = call;
        call = Call(is_pure ? invoke_pure_closure_op_ : invoke_closure_op_,
                    {var, Tuple(call->args)}, {}, {orig_sinfo});
      }
    }

    if (auto opt_var = call->op.as<Var>()) {
      auto var = opt_var.value();
      if (auto it = nested_closure_map_.find(var); it != nested_closure_map_.end()) {
        Call nested_call = it->second;

        Array<relay::Expr> new_args = call->args;
        for (const auto arg : nested_call->args) {
          new_args.push_back(arg);
        }

        auto prev = call;
        call = Call(nested_call->op, new_args, call->attrs, call->sinfo_args);
      }
    }

    return ExprMutator::VisitExpr_(call.get());
  }

  Expr VisitExpr_(const VarNode* op) override {
    auto var = GetRef<Var>(op);
    if (auto it = rebind_map_.find(var); it != rebind_map_.end()) {
      return it->second;
    }
    return ExprMutator::VisitExpr_(op);
  }

  bool IsClosure(Expr val) {
    if (auto opt_var = val.as<Var>()) {
      if (closures_.count(opt_var.value())) {
        return true;
      }
      if (auto bound_value = builder_->LookupBinding(opt_var.value())) {
        val = bound_value.value();
      }
    }

    if (const auto* call_node = val.as<relax::CallNode>()) {
      // recursive call
      auto op = call_node->op;
      if (auto local_var = op.as<Var>()) {
        return IsClosure(local_var.value());
      } else if (auto global_var = op.as<GlobalVar>()) {
        return IsClosure(global_var.value());
      } else {
        return make_closure_op_ == op;
      }

    } else if (const auto* global_var = val.as<GlobalVarNode>()) {
      if (closures_.count(GetRef<GlobalVar>(global_var))) {
        return true;
      }
      IRModule ctx_mod = builder_->GetContextIRModule();
      ICHECK(ctx_mod->functions.size() > 0);
      BaseFunc func = ctx_mod->Lookup(GetRef<GlobalVar>(global_var));
      const auto* func_node = func.as<FunctionNode>();
      if (func_node) {
        return IsClosure(func_node->body);
      } else {
        return false;
      }

    } else if (const auto* func_node = val.as<FunctionNode>()) {
      return IsClosure(func_node->body);

    } else if (const auto* seq_node = val.as<SeqExprNode>()) {
      return IsClosure(seq_node->body);

    } else {
      return false;
    }
  }

  IRModule Lift() {
    auto glob_funcs = mod_->functions;
    for (auto [gvar, base_func] : glob_funcs) {
      if (auto opt = base_func.as<Function>()) {
        // Must visit the function itself, and not just the function
        // body, to ensure that EraseToWellDefined recognized symbolic
        // variables that are exposed by the function signature.
        auto func = Downcast<Function>(VisitExpr(opt.value()));
        builder_->UpdateFunction(gvar, func);
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  std::unordered_map<Var, Call, ObjectPtrHash, ObjectPtrEqual> nested_closure_map_;
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> rebind_map_;
  std::unordered_set<Variant<GlobalVar, Var>, ObjectPtrHash, ObjectPtrEqual> closures_;
  Optional<Var> current_lambda_var_ = NullOpt;
  IRModule mod_;

  std::unordered_map<const FunctionNode*, String> lifted_names_;

  /*! \brief Cache ops that would be used later to reduce lookup overhead. */
  const Op& make_closure_op_ = Op::Get("relax.make_closure");
  const Op& invoke_closure_op_ = Op::Get("relax.invoke_closure");
  const Op& invoke_pure_closure_op_ = Op::Get("relax.invoke_pure_closure");
};

namespace transform {

Pass LambdaLift() {
  auto pass_func = [=](IRModule mod, PassContext pc) { return relax::LambdaLifter(mod).Lift(); };
  return tvm::transform::CreateModulePass(pass_func, 1, "LambdaLift", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LambdaLift").set_body_typed(LambdaLift);

}  // namespace transform
}  // namespace relax
}  // namespace tvm

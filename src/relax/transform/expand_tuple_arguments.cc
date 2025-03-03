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

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <algorithm>
#include <tuple>

namespace tvm {
namespace relax {

namespace {

template <typename T, typename U>
using PMap = std::unordered_map<T, U, ObjectPtrHash, ObjectPtrEqual>;

Optional<Function> ExpandParams(Function func) {
  bool is_exposed = func->attrs.GetAttr<String>(tvm::attr::kGlobalSymbol).defined();
  if (is_exposed) return NullOpt;

  bool has_tuple_param = std::any_of(
      func->params.begin(), func->params.end(),
      [](const Var& param) -> bool { return param->struct_info_.as<TupleStructInfoNode>(); });

  if (!has_tuple_param) return NullOpt;

  Array<Var> params;
  Array<Binding> bindings;

  std::function<void(const Var&)> expand_param = [&](const Var& param) {
    if (auto sinfo = param->struct_info_.as<TupleStructInfoNode>()) {
      Array<Expr> internal_tuple;
      for (size_t i = 0; i < sinfo->fields.size(); i++) {
        auto name = static_cast<const std::stringstream&>(std::stringstream()
                                                          << param->name_hint() << "_" << i)
                        .str();
        Var new_param(name, sinfo->fields[i]);
        internal_tuple.push_back(new_param);
        expand_param(new_param);
      }
      bindings.push_back(VarBinding(param, Tuple(internal_tuple)));
    } else {
      params.push_back(param);
    }
  };

  for (const auto& param : func->params) {
    expand_param(param);
  }

  FuncStructInfo new_sinfo(params.Map([](const auto& var) { return GetStructInfo(var); }),
                           func->ret_struct_info,
                           Downcast<FuncStructInfo>(func->struct_info_)->purity);

  auto write_ptr = func.CopyOnWrite();
  write_ptr->params = params;
  write_ptr->body = SeqExpr({BindingBlock(bindings)}, func->body);
  write_ptr->struct_info_ = new_sinfo;

  return func;
}

class TupleExpander : public ExprMutator {
 public:
  explicit TupleExpander(PMap<GlobalVar, GlobalVar> callees) : replacements_(callees) {}

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* op) override {
    auto node = Downcast<Call>(ExprMutator::VisitExpr_(op));

    if (auto gvar = node->op.as<GlobalVar>()) {
      if (auto it = replacements_.find(gvar.value()); it != replacements_.end()) {
        Array<Expr> new_args;

        std::function<void(const Expr&)> expand_arg = [&](const Expr& arg) {
          if (auto sinfo = arg->struct_info_.as<TupleStructInfoNode>()) {
            for (size_t i = 0; i < sinfo->fields.size(); i++) {
              expand_arg(TupleGetItem(arg, i));
            }
          } else {
            new_args.push_back(arg);
          }
        };

        for (const auto& arg : node->args) {
          expand_arg(arg);
        }

        auto write_ptr = node.CopyOnWrite();
        write_ptr->op = it->second;
        write_ptr->args = new_args;
      }
    }

    return node;
  }

  PMap<GlobalVar, GlobalVar> replacements_;
};

}  // namespace

namespace transform {

Pass ExpandTupleArguments() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) -> IRModule {
    PMap<GlobalVar, GlobalVar> gvar_replacements;

    {
      PMap<GlobalVar, Function> new_callees;

      for (const auto& [gvar, base_func] : mod->functions) {
        if (auto func = base_func.as<Function>()) {
          if (auto opt = ExpandParams(func.value())) {
            auto new_func = opt.value();
            GlobalVar new_gvar(gvar->name_hint, new_func->checked_type_);
            new_gvar->struct_info_ = new_func->struct_info_;
            gvar_replacements[gvar] = new_gvar;
            new_callees[new_gvar] = new_func;
          }
        }
      }

      if (gvar_replacements.empty()) {
        return mod;
      }
      auto write_ptr = mod.CopyOnWrite();
      for (auto [old_gvar, new_gvar] : gvar_replacements) {
        write_ptr->Remove(old_gvar);
        write_ptr->Add(new_gvar, new_callees.at(new_gvar));
      }
    }

    TupleExpander mutator(std::move(gvar_replacements));

    IRModule caller_updates;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto mutated = Downcast<Function>(mutator.VisitExpr(func.value()));
        if (!mutated.same_as(base_func)) {
          caller_updates->Add(gvar, mutated);
        }
      }
    }

    if (caller_updates->functions.size()) {
      mod.CopyOnWrite()->Update(caller_updates);
    }
    return mod;
  };
  auto inner_pass = CreateModulePass(pass_func, 0, "ExpandTupleArgumentsInner", {});

  return tvm::transform::Sequential(
      {
          inner_pass,
          CanonicalizeBindings(),
          DeadCodeElimination({}),
      },
      "ExpandTupleArguments");
}

TVM_REGISTER_GLOBAL("relax.transform.ExpandTupleArguments").set_body_typed(ExpandTupleArguments);

}  // namespace transform

}  // namespace relax
}  // namespace tvm

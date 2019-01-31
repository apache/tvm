/*!
 * Copyright (c) 2019 by Contributors
 * \file src/relay/pass/optimizer.cc
 * \brief Relay optimizer implementation.
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/optimizer.h>

namespace tvm {
namespace relay {
namespace optimize {

using tvm::IRPrinter;

PassState PassStateNode::make(Module mod) {
  auto n = make_node<PassStateNode>();
  n->mod = std::move(mod);
  return PassState(n);
}

ModulePass ModulePassNode::make(std::string name, int opt_level,
                                PassFunc<Module> pass_func) {
  auto n = make_node<ModulePassNode>();
  n->name = std::move(name);
  n->opt_level = std::move(opt_level);
  n->pass_kind = PassKind::kModuleKind;
  n->pass_func = std::move(pass_func);
  return ModulePass(n);
}

// PassState: Module -> Module
void ModulePassNode::run(PassState* state) const {
  LOG(INFO) << "Executing module pass : " << this->name
            << " with opt level: " << opt_level << "\n";
  const auto& pass_st_node = (*state).operator->();
  CHECK(pass_st_node != nullptr);
  auto foreach = pass_func(*state);
  Module updated_module = foreach(pass_st_node->mod);
  CHECK(updated_module.defined());
  // Update pass state
  *state = PassStateNode::make(updated_module);;
}

FunctionPass FunctionPassNode::make(std::string name, int opt_level,
                                    PassFunc<Function> pass_func) {
  auto n = make_node<FunctionPassNode>();
  n->name = std::move(name);
  n->opt_level = std::move(opt_level);
  n->pass_kind = PassKind::kFunctionKind;
  n->pass_func = std::move(pass_func);
  return FunctionPass(n);
}

// PassState: Function -> Function
void FunctionPassNode::run(PassState* state) const {
  LOG(INFO) << "Executing function pass : " << this->name
            << " with opt level: " << this->opt_level << "\n";
  const auto pass_st_node = (*state).operator->();
  CHECK(pass_st_node != nullptr);
  auto foreach = pass_func(*state);
  ModuleNode* mod = pass_st_node->mod.operator->();
  std::vector<std::pair<GlobalVar, Function>> updated_funcs;
  for (const auto& it : mod->functions) {
    if (!SkipFunction(it.second)) {
      auto updated_func = foreach(it.second);
      CHECK(updated_func.defined());
      updated_funcs.push_back({std::move(it.first), std::move(updated_func)});
    }
  }

  // Update the optimized functions.
  for (const auto& it : updated_funcs) {
    mod->Update(it.first, it.second);
  }
  *state = PassStateNode::make(GetRef<Module>(mod));
}

// TODO(zhiics) Create an enum attribute for FunctionNode
// enum Attribute {kPrimitive, kSkipOptimization}
bool FunctionPassNode::SkipFunction(const Function& func) const {
  NodeRef res = FunctionGetAttr(func, "SkipOptimization");
  const ir::IntImm* pval = res.as<ir::IntImm>();
  return pval && pval->value != 0;
}

ExprPass ExprPassNode::make(std::string name, int opt_level,
                            PassFunc<Expr> pass_func) {
  auto n = make_node<ExprPassNode>();
  n->name = std::move(name);
  n->opt_level = std::move(opt_level);
  n->pass_kind = PassKind::kExprKind;
  n->pass_func = std::move(pass_func);
  return ExprPass(n);
}

// PassState: Expr -> Expr
void ExprPassNode::run(PassState* state) const {
  LOG(INFO) << "Executing Expr pass on PassState: " << this->name << "\n";
  const auto pass_st_node = (*state).operator->();
  CHECK(pass_st_node != nullptr);
  auto foreach = pass_func(*state);
  ModuleNode* mod = pass_st_node->mod.operator->();
  std::vector<std::pair<GlobalVar, Function>> updated_funcs;
  for (const auto& it : mod->functions) {
    const auto& fn = it.second.operator->();
    auto updated_body = foreach(fn->body);
    CHECK(updated_body.defined());
    auto new_func = FunctionNode::make(fn->params, updated_body, fn->ret_type,
                                       fn->type_params, fn->attrs);
    updated_funcs.push_back({std::move(it.first), std::move(new_func)});
  }

  // Update the optimized functions.
  for (const auto& it : updated_funcs) {
    mod->Update(it.first, it.second);
  }
  *state = PassStateNode::make(GetRef<Module>(mod));
}

void Optimizer::Optimize() const {
  for (const Pass& pass : passes_) {
    CHECK(pass.defined()) << "Found undefined pass for optimization.";
    pass->run(&state_);
  }
}

void Optimize(const tvm::Array<Pass>& passes, PassState* state) {
  LOG(INFO) << "Start executing optimization passes." << "\n";
  Optimizer pm(*state, passes);
  pm.Optimize();
  *state = pm.state_;
}

TVM_REGISTER_NODE_TYPE(ModulePassNode);

TVM_REGISTER_API("relay._optimize.ModulePass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string pass_name = args[0];
  int opt_level = args[1];
  PackedFunc py_pass_func = args[2];
  PassFunc<Module> pass_func = [py_pass_func](const PassState& state) {
    PackedFunc py_for_each = py_pass_func(state);
    return [py_for_each](Module m) {
      Module r = py_for_each(m);
      CHECK(r.defined());
      return r;
    };
  };
  *ret = ModulePassNode::make(pass_name, opt_level, pass_func);
});

TVM_REGISTER_API("relay._optimize.RunModulePass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  ModulePass pass = args[0];
  PassState state = args[1];
  CHECK(pass.defined())
      << "Running a pass on undefined ExprPass is not allowed."
      << "\n";
  pass->run(&state);
  *ret = state;
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ModulePassNode>([](const ModulePassNode* node,
                                 tvm::IRPrinter* p) {
  p->stream << "Run Module pass: " << node->name
            << " at the optimization level " << node->opt_level;
});

TVM_REGISTER_NODE_TYPE(FunctionPassNode);

TVM_REGISTER_API("relay._optimize.FunctionPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string pass_name = args[0];
  int opt_level = args[1];
  PackedFunc py_pass_func = args[2];
  PassFunc<Function> pass_func = [py_pass_func](const PassState& state) {
    PackedFunc py_for_each = py_pass_func(state);
    return [py_for_each](Function i) {
      Function r = py_for_each(i);
      CHECK(r.defined());
      return r;
    };
  };
  *ret = FunctionPassNode::make(pass_name, opt_level, pass_func);
});

TVM_REGISTER_API("relay._optimize.RunFunctionPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  FunctionPass pass = args[0];
  PassState state = args[1];
  CHECK(pass.defined())
      << "Running a pass on undefined ExprPass is not allowed."
      << "\n";
  pass->run(&state);
  *ret = state;
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<FunctionPassNode>([](const FunctionPassNode* node,
                                   tvm::IRPrinter* p) {
  p->stream << "Run Function pass: " << node->name
            << " at the optimization level " << node->opt_level;
});

TVM_REGISTER_NODE_TYPE(ExprPassNode);

TVM_REGISTER_API("relay._optimize.ExprPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string pass_name = args[0];
  int opt_level = args[1];
  PackedFunc py_pass_func = args[2];
  PassFunc<Expr> pass_func = [py_pass_func](const PassState& state) {
    PackedFunc py_for_each = py_pass_func(state);
    return [py_for_each](Expr i) {
      Expr r = py_for_each(i);
      CHECK(r.defined());
      return r;
    };
  };
  *ret = ExprPassNode::make(pass_name, opt_level, pass_func);
});

TVM_REGISTER_API("relay._optimize.RunExprPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  ExprPass pass = args[0];
  CHECK(pass.defined())
      << "Running a pass on undefined ExprPass is not allowed."
      << "\n";
  PassState state = args[1];
  pass->run(&state);
  *ret = state;
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ExprPassNode>([](const ExprPassNode* node,
                               tvm::IRPrinter* p) {
  p->stream << "Run Expr pass: " << node->name
            << " at the optimization level " << node->opt_level;
});

TVM_REGISTER_NODE_TYPE(PassStateNode);

TVM_REGISTER_API("relay._optimize.PassState")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Module mod = args[0];
  *ret = PassStateNode::make(mod);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PassStateNode>([](const PassStateNode* node,
                                tvm::IRPrinter* p) {
  if (node->mod.defined()) {
    p->stream << "Pass state with module: " << "\n";
    p->stream << RelayPrint(node->mod);
  } else {
    p->stream << "Skip printing as no module is defined in the pass state.";
  }
});

TVM_REGISTER_API("relay._optimize.Optimize")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  tvm::Array<Pass> passes = args[0];
  PassState state = args[1];
  Optimize(passes, &state);
  *ret = state;
});

}  // namespace optimize
}  // namespace relay
}  // namespace tvm

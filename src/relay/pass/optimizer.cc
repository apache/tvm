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

PassContext PassContextNode::make() {
  auto ctx = make_node<PassContextNode>();
  return PassContext(ctx);
}

std::vector<std::string> PassNode::RequiredPasses() const {
  std::vector<std::string> dependent;
  for (const auto& it : required_passes) {
    dependent.push_back(it.as<tvm::ir::StringImm>()->value);
  }
  return dependent;
}

ModulePass ModulePassNode::make(std::string name, int opt_level,
                                PassFunc<Module> pass_func, bool enabled,
                                tvm::Array<tvm::Expr> required_passes) {
  auto n = make_node<ModulePassNode>();
  n->name = std::move(name);
  n->opt_level = std::move(opt_level);
  n->pass_kind = PassKind::kModuleKind;
  n->pass_func = std::move(pass_func);
  n->enabled = std::move(enabled);
  n->required_passes = std::move(required_passes);
  return ModulePass(n);
}

// Module -> Module optimizations.
// TODO(zhiics) 1. Check and handle the required passes.
//              2. Probably use CoW for all places that use module instead of
//              returning the updated one.
Module ModulePassNode::Run(const Module& mod,
                           const PassContext& pass_ctx) const {
  LOG(INFO) << "Executing module pass : " << this->name
            << " with opt level: " << opt_level << "\n";
  CHECK(mod.defined());
  auto foreach = pass_func(mod);
  auto updated_mod = foreach(mod);
  CHECK(updated_mod.defined());
  return std::move(updated_mod);
}

FunctionPass FunctionPassNode::make(std::string name, int opt_level,
                                    PassFunc<Function> pass_func, bool enabled,
                                    tvm::Array<tvm::Expr> required_passes) {
  auto n = make_node<FunctionPassNode>();
  n->name = std::move(name);
  n->opt_level = std::move(opt_level);
  n->pass_kind = PassKind::kFunctionKind;
  n->pass_func = std::move(pass_func);
  n->enabled = std::move(enabled);
  n->required_passes = std::move(required_passes);
  return FunctionPass(n);
}

// Perform Module -> Module optimizations at the Function level.
// TODO(zhiics) Check and handle the required passes.
Module FunctionPassNode::Run(const Module& mod,
                             const PassContext& pass_ctx) const {
  LOG(INFO) << "Executing function pass : " << this->name
            << " with opt level: " << this->opt_level << "\n";
  CHECK(mod.defined());
  auto foreach = pass_func(mod);
  std::vector<std::pair<GlobalVar, Function>> updated_funcs;
  ModuleNode* mod_node = mod.operator->();
  for (const auto& it : mod_node->functions) {
    if (!SkipFunction(it.second)) {
      auto updated_func = foreach(it.second);
      CHECK(updated_func.defined());
      updated_funcs.push_back({std::move(it.first), std::move(updated_func)});
    }
  }

  // Update the optimized functions.
  for (const auto& it : updated_funcs) {
    mod_node->Update(it.first, it.second);
  }

  return GetRef<Module>(mod_node);
}

// TODO(zhiics) Create an enum attribute for FunctionNode
// enum Attribute {kPrimitive, kSkipOptimization}
bool FunctionPassNode::SkipFunction(const Function& func) const {
  NodeRef res = FunctionGetAttr(func, "SkipOptimization");
  const ir::IntImm* pval = res.as<ir::IntImm>();
  return pval && pval->value != 0;
}

Module Optimizer::Optimize() {
  for (const Pass& pass : passes_) {
    CHECK(pass.defined()) << "Found undefined pass for optimization.";
    module_ = pass->Run(module_, pass_ctx_);
  }
  return module_;
}

Module Optimize(const tvm::Array<Pass>& passes,
                const Module& mod,
                const PassContext& pass_ctx) {
  LOG(INFO) << "Start executing optimization passes." << "\n";
  Optimizer pm(mod, passes, pass_ctx);
  pm.Optimize();
  return pm.module_;
}

TVM_REGISTER_NODE_TYPE(ModulePassNode);

TVM_REGISTER_API("relay._optimize.ModulePass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string pass_name = args[0];
  int opt_level = args[1];
  PackedFunc py_pass_func = args[2];
  bool enabled = args[3];
  Array<tvm::Expr> required_passes = args[4];
  PassFunc<Module> pass_func = [py_pass_func](const Module& mod) {
    PackedFunc py_for_each = py_pass_func(mod);
    return [py_for_each](Module m) {
      Module r = py_for_each(m);
      CHECK(r.defined());
      return r;
    };
  };
  *ret = ModulePassNode::make(pass_name, opt_level, pass_func, enabled,
                              required_passes);
});

TVM_REGISTER_API("relay._optimize.RunModulePass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  ModulePass pass = args[0];
  Module mod = args[1];
  PassContext pass_ctx = args[2];
  CHECK(pass.defined())
      << "Running a pass on undefined ModulePass is not allowed."
      << "\n";
  *ret = pass->Run(mod, pass_ctx);
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
  bool enabled = args[3];
  Array<tvm::Expr> required_passes = args[4];
  PassFunc<Function> pass_func = [py_pass_func](const Module& mod) {
    PackedFunc py_for_each = py_pass_func(mod);
    return [py_for_each](Function i) {
      Function r = py_for_each(i);
      CHECK(r.defined());
      return r;
    };
  };
  *ret = FunctionPassNode::make(pass_name, opt_level, pass_func, enabled,
                                required_passes);
});

TVM_REGISTER_API("relay._optimize.RunFunctionPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  FunctionPass pass = args[0];
  Module mod = args[1];
  PassContext pass_ctx = args[2];
  CHECK(pass.defined())
      << "Running a pass on undefined ModulePass is not allowed."
      << "\n";
  *ret = pass->Run(mod, pass_ctx);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<FunctionPassNode>([](const FunctionPassNode* node,
                                   tvm::IRPrinter* p) {
  p->stream << "Run Function pass: " << node->name
            << " at the optimization level " << node->opt_level;
});

TVM_REGISTER_NODE_TYPE(PassContextNode);

TVM_REGISTER_API("relay._optimize.PassContext")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = PassContextNode::make();
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PassContextNode>([](const PassContextNode* node,
                                tvm::IRPrinter* p) {
    p->stream << "TODO(zhiics): printing context";
});

TVM_REGISTER_API("relay._optimize.Optimize")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  tvm::Array<Pass> passes = args[0];
  Module mod = args[1];
  PassContext pass_ctx = args[2];
  *ret = Optimize(passes, mod, pass_ctx);
});

}  // namespace optimize
}  // namespace relay
}  // namespace tvm

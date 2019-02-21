/*!
 * Copyright (c) 2019 by Contributors
 * \file src/relay/pass/pass_manager.cc
 * \brief Relay pass manager implementation.
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pass_manager.h>

namespace tvm {
namespace relay {
namespace pass {

using tvm::IRPrinter;

PassContext PassContextNode::make() {
  auto ctx = make_node<PassContextNode>();
  return PassContext(ctx);
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

// Module -> Module optimizations.
// TODO(zhiics) 1. Check and handle the required passes.
//              2. Probably use CoW for all places that use module instead of
//              returning the updated one.
Module ModulePassNode::Run(const Module& mod) const {
  LOG(INFO) << "Executing module pass : " << this->name
            << " with opt level: " << opt_level << "\n";
  CHECK(mod.defined());
  auto foreach = pass_func(mod);
  auto updated_mod = foreach(mod);
  CHECK(updated_mod.defined());
  return updated_mod;
}

std::vector<std::string> ModulePassNode::Required() const {
  // TODO(zhiics) Return required passes based on the current pass info.
  std::vector<std::string> required;
  return required;
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

// Perform Module -> Module optimizations at the Function level.
// TODO(zhiics) Check and handle the required passes.
Module FunctionPassNode::Run(const Module& mod) const {
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

std::vector<std::string> FunctionPassNode::Required() const {
  // TODO(zhiics) Return required passes based on the current pass info.
  std::vector<std::string> required;
  return required;
}

SequentialPass SequentialPassNode::make(std::string name, int opt_level,
                                        tvm::Array<Pass> passes,
                                        tvm::Array<tvm::Expr> disabled) {
  auto n = make_node<SequentialPassNode>();
  n->name = std::move(name);
  n->opt_level = std::move(opt_level);
  n->pass_kind = PassKind::kSequentialKind;
  n->passes = std::move(passes);
  n->disabled = std::move(disabled);
  return SequentialPass(n);
}

Module SequentialPassNode::Run(const Module& module) const {
  Module mod = module;
  for (const Pass& pass : passes) {
    CHECK(pass.defined()) << "Found undefined pass for optimization.";
    mod = pass->Run(mod);
  }
  return mod;
}

std::vector<std::string> SequentialPassNode::Required() const {
  // TODO(zhiics) Return required passes based on the current pass info.
  std::vector<std::string> required;
  return required;
}

void SequentialPassNode::ResolveDependency(const Module& mod) {
  // TODO(zhiics) Implement it.
  // 1. Consider the required passes for each pass.
  // 2. Only resolve the enabled passes.
  // 3. Build a dependency graph. Probably we need to update the pass list.
}

std::vector<std::string> SequentialPassNode::DisabledPasses() const {
  std::vector<std::string> ret;
  for (const auto& it : disabled) {
    const auto* str = it.as<tvm::ir::StringImm>();
    CHECK(str) << "disabled passes must be string.";
    ret.push_back(str->value);
  }
  return ret;
}

ModulePass CreateModulePass(const std::string& name, int opt_level,
                            const PassFunc<Module>& pass_func) {
  return ModulePassNode::make(name, opt_level, pass_func);
}

FunctionPass CreateFunctionPass(const std::string& name, int opt_level,
                                const PassFunc<Function>& pass_func) {
  return FunctionPassNode::make(name, opt_level, pass_func);
}

SequentialPass CreateSequentialPass(const std::string& name, int opt_level,
                                    const tvm::Array<Pass>& passes,
                                    const tvm::Array<tvm::Expr>& disabled) {
  return SequentialPassNode::make(name, opt_level, passes, disabled);
}

TVM_REGISTER_NODE_TYPE(ModulePassNode);

TVM_REGISTER_API("relay._ir_pass.CreateModulePass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string pass_name = args[0];
  int opt_level = args[1];
  PackedFunc py_pass_func = args[2];
  PassFunc<Module> pass_func = [py_pass_func](const Module& mod) {
    PackedFunc py_for_each = py_pass_func(mod);
    return [py_for_each](Module m) {
      Module r = py_for_each(m);
      CHECK(r.defined());
      return r;
    };
  };
  *ret = CreateModulePass(pass_name, opt_level, pass_func);
});

TVM_REGISTER_API("relay._ir_pass.RunModulePass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  ModulePass pass = args[0];
  Module mod = args[1];
  CHECK(pass.defined())
      << "Running a pass on undefined ModulePass is not allowed."
      << "\n";
  *ret = pass->Run(mod);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ModulePassNode>([](const ModulePassNode* node,
                                 tvm::IRPrinter* p) {
  p->stream << "Run Module pass: " << node->name
            << " at the optimization level " << node->opt_level;
});

TVM_REGISTER_NODE_TYPE(FunctionPassNode);

TVM_REGISTER_API("relay._ir_pass.CreateFunctionPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string pass_name = args[0];
  int opt_level = args[1];
  PackedFunc py_pass_func = args[2];
  PassFunc<Function> pass_func = [py_pass_func](const Module& mod) {
    PackedFunc py_for_each = py_pass_func(mod);
    return [py_for_each](Function i) {
      Function r = py_for_each(i);
      CHECK(r.defined());
      return r;
    };
  };
  *ret = CreateFunctionPass(pass_name, opt_level, pass_func);
});

TVM_REGISTER_API("relay._ir_pass.RunFunctionPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  FunctionPass pass = args[0];
  Module mod = args[1];
  CHECK(pass.defined())
      << "Running a pass on undefined ModulePass is not allowed."
      << "\n";
  *ret = pass->Run(mod);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<FunctionPassNode>([](const FunctionPassNode* node,
                                   tvm::IRPrinter* p) {
  p->stream << "Run Function pass: " << node->name
            << " at the optimization level " << node->opt_level;
});

TVM_REGISTER_NODE_TYPE(SequentialPassNode);

TVM_REGISTER_API("relay._ir_pass.CreateSequentialPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string pass_name = args[0];
  int opt_level = args[1];
  tvm::Array<Pass> passes = args[2];
  tvm::Array<tvm::Expr> disabled = args[3];
  *ret = SequentialPassNode::make(pass_name, opt_level, passes, disabled);
});

TVM_REGISTER_API("relay._ir_pass.RunSequentialPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  SequentialPass pass = args[0];
  Module mod = args[1];
  CHECK(pass.defined())
      << "Running passes on undefined SequentialPass is not allowed."
      << "\n";
  *ret = pass->Run(mod);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<SequentialPassNode>([](const SequentialPassNode* node,
                                     tvm::IRPrinter* p) {
  p->stream << "Run SequentialPass pass: " << node->name
            << " at the optimization level. " << node->opt_level;
  p->stream << "The passes will be executed are: [";
  for (const auto& it : node->passes) {
    p->stream << it.operator->()->name << " ";
  }
  p->stream << "]";
});

TVM_REGISTER_API("relay._ir_pass.SetContext")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Pass pass = args[0];
  PassContext pass_ctx = args[1];
  pass->SetContext(pass_ctx);
});

TVM_REGISTER_NODE_TYPE(PassContextNode);

TVM_REGISTER_API("relay._ir_pass.PassContext")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = PassContextNode::make();
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PassContextNode>([](const PassContextNode* node,
                                tvm::IRPrinter* p) {
    p->stream << "TODO(zhiics): printing context";
});

}  // namespace pass
}  // namespace relay
}  // namespace tvm

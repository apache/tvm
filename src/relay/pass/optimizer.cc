/*!
 * Copyright (c) 2019 by Contributors
 * \file src/relay/pass/optimizer.cc
 * \brief Relay optimizer implementation.
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/optimizer.h>
#include <tvm/relay/type.h>

namespace tvm {
namespace relay {

using tvm::IRPrinter;

template <typename T>
FunctionPass<T> FunctionPassNode<T>::make(std::string name,
                                          PassFunc<T> pass_func) {
  auto n = make_node<FunctionPassNode<T>>();
  n->name = std::move(name);
  n->pass_kind = PassKind::kFunctionKind;
  n->pass_func = std::move(pass_func);
  return FunctionPass<T>(n);
}

TVM_REGISTER_API("relay._make.FunctionPass")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  std::string pass_name = args[0];
  PackedFunc py_pass_func = args[1];
  PassFunc<Module> pass_func = [py_pass_func](const Module& mod) {
    PackedFunc py_for_each = py_pass_func(mod);
    return [py_for_each](Function i) {
      Function r = py_for_each(i);
      CHECK(r.defined());
      return r;
    };
  };
  *ret = FunctionPassNode<Module>::make(pass_name, pass_func);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<FunctionPassNode<Module>>(
  [](const FunctionPassNode<Module>* node, tvm::IRPrinter* p) {
  p->stream << "FunctionPass(TODO)";
});

template <>
void FunctionPassNode<Module>::run(const Module& unit) const {
  std::cout << "Executing function Pass on Module: " << this->name << std::endl;
  auto foreach = this->pass_func(unit);
  for (const auto& it : unit.operator->()->functions) {
    auto updated_item = foreach(it.second);
    CHECK(updated_item.defined());
    unit->Update(it.first, updated_item);
  }
}

template <>
void FunctionPassNode<PassState>::run(const PassState& unit) const {
  std::cout << "Executing function Pass on PassState: " << this->name
            << std::endl;
  auto foreach = this->pass_func(unit);
  ModuleNode* mod = unit.operator->()->mod.operator->();
  for (const auto& it : mod->functions) {
    auto updated_item = foreach(it.second);
    CHECK(updated_item.defined());
    mod->Update(it.first, updated_item);
  }
}

// TODO(zhiics) Create an enum attribute for FunctionNode
// enum Attribute {kPrimitive, kSkipOptimization}
template <typename T>
bool FunctionPassNode<T>::SkipFunction(const Function& func) const {
  NodeRef res = FunctionGetAttr(func, "SkipOptimization");
  const ir::IntImm* pval = res.as<ir::IntImm>();
  return pval && pval->value != 0;
}

template <typename T>
ExprPass<T> ExprPassNode<T>::make(std::string name,
                            PassFunc<T, Expr, Expr> pass_func) {
  NodePtr<ExprPassNode<T>> n = make_node<ExprPassNode<T>>();
  n->name = std::move(name);
  n->pass_kind = PassKind::kExprKind;
  n->pass_func = std::move(pass_func);
  return ExprPass<T>(n);
}

TVM_REGISTER_API("relay._make.ExprPass")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  std::string pass_name = args[0];
  PackedFunc py_pass_func = args[1];
  PassFunc<Module, Expr, Expr> pass_func = [py_pass_func](const Module& mod) {
    PackedFunc py_for_each = py_pass_func(mod);
    return [py_for_each](Expr i) {
      Expr r = py_for_each(i);
      CHECK(r.defined());
      return r;
    };
  };
  *ret = ExprPassNode<Module>::make(pass_name, pass_func);
});

template <typename T>
void ExprPassNode<T>::run(const Module& mod) const {
  std::cout << "Executing Expr Pass: " << this->name << std::endl;
  // TODO(zhiics) what are the target expression for optimization?
  // function body?
}

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ExprPassNode<Module>>([](const ExprPassNode<Module>* node,
                                       const tvm::IRPrinter* p) {
  p->stream << "ExprPass(TODO) what to print?";
});

PassState PassStateNode::make(Module mod, GlobalVar current_func) {
  auto node = make_node<PassStateNode>();
  node->mod = std::move(mod);
  node->current_func = current_func;
  return PassState(node);
}

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PassStateNode>([](const PassStateNode* node,
                                const tvm::IRPrinter* p) {
  p->stream << "PassState(TODO)";
});

PassManager PassManagerNode::make(tvm::Array<Pass> passes) {
  auto node = make_node<PassManagerNode>();
  node->passes = std::move(passes);
  return PassManager(node);
}

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PassManagerNode>([](const PassManagerNode* node,
                                  const tvm::IRPrinter* p) {
  p->stream << "PassManager(TODO)";
});

void Optimize(const Module& mod, tvm::Array<Pass> passes) {
  std::cout << "Optimize" << std::endl;
  PassManager pm = PassManagerNode::make(passes);
  for (const Pass& pass : pm.operator->()->passes) {
    pass->run(mod);
  }
}

TVM_REGISTER_API("relay._make.PassManager")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = PassManagerNode::make(args[0]);
});

TVM_REGISTER_API("relay._opt.optimize")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  tvm::Array<Pass> passes = args[1];
  Optimize(args[0], passes);
});

}  // namespace relay
}  // namespace tvm

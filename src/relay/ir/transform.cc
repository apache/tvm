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
 * \file relay/ir/transform.cc
 * \brief Relay specific transformation passes.
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/node/repr_printer.h>
#include <tvm/relay/transform.h>


namespace tvm {
namespace relay {
namespace transform {

class FunctionPass;

/*!
 * \brief Function-level passes are used to implement various global
 * optimizations for a given Relay module. It fetches one function at a time
 * from the function list in the module for optimization.
 *
 * Note that the scope of passes at this level is a Relay function. Therefore,
 * we cannot add or delete a function through these passes as they are not aware
 * of the global information.
 */
class FunctionPassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The packed pass function sketches the real optimization. For
   * instance, we can implement a pass that works on a Relay function as a
   * `pass_func` and let it run on a given module. The same `pass_func` will
   * then be applied on each function in the module.
   */
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func;

  FunctionPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pass_info", &pass_info);
  }

  /*!
   * \brief Run a function pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param mod The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(const IRModule& mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  TVM_DLL static FunctionPass make(
      runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func,
      PassInfo pass_info);

  static constexpr const char* _type_key = "relay.FunctionPass";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionPassNode, PassNode);

 private:
  /*
   * \brief Check if a function should be skipped for optimization.
   *
   * \param func The target function to be checked.
   *
   * \return Return true if the function will be skipped, otherwise false.
   */
  bool SkipFunction(const Function& func) const;
};

class FunctionPass : public Pass {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(FunctionPass, Pass, FunctionPassNode);
};

FunctionPass FunctionPassNode::make(
    runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func,
    PassInfo pass_info) {
  auto n = make_object<FunctionPassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  return FunctionPass(n);
}

// Perform Module -> Module optimizations at the Function level.
IRModule FunctionPassNode::operator()(const IRModule& mod,
                                      const PassContext& pass_ctx) const {
  const PassInfo& pass_info = Info();
  CHECK(mod.defined());
  DLOG(INFO) << "Executing function pass : "
             << pass_info->name
             << " with opt level: "
             << pass_info->opt_level;

  // Execute the pass function and return a new module.
  IRModule updated_mod = IRModule(mod->functions, mod->type_definitions, mod->Imports());
  std::vector<std::pair<GlobalVar, Function> > updates;
  for (const auto& it : updated_mod->functions) {
    // only picks up relay::Function
    if (auto* n = it.second.as<FunctionNode>()) {
      Function func = GetRef<Function>(n);
      auto updated_func = SkipFunction(func)
                          ? func
                          : pass_func(func, updated_mod, pass_ctx);
      updates.push_back({it.first, updated_func});
    }
  }

  for (const auto& pair : updates) {
    updated_mod->Add(pair.first, pair.second, true);
  }
  return updated_mod;
}

bool FunctionPassNode::SkipFunction(const Function& func) const {
  ObjectRef skip_opt = FunctionGetAttr(func, attr::kSkipOptimization);
  const tir::IntImmNode* pval = skip_opt.as<tir::IntImmNode>();
  return (pval && pval->value != 0) || (!func->UseDefaultCompiler());
}

Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
    int opt_level,
    const std::string& name,
    const tvm::Array<tvm::PrimExpr>& required) {
  PassInfo pass_info = PassInfo(opt_level, name, required);
  return FunctionPassNode::make(pass_func, pass_info);
}

TVM_REGISTER_NODE_TYPE(FunctionPassNode);

TVM_REGISTER_GLOBAL("relay._transform.MakeFunctionPass")
.set_body_typed(FunctionPassNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<FunctionPassNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const FunctionPassNode*>(ref.get());
  const PassInfo info = node->Info();
  p->stream << "Run Function pass: " << info->name
            << " at the optimization level " << info->opt_level;
});

}  // namespace transform
}  // namespace relay
}  // namespace tvm

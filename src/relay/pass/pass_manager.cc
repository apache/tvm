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
 * Copyright (c) 2019 by Contributors
 * \file src/relay/pass/pass_manager.cc
 * \brief Relay pass manager implementation.
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pass.h>

namespace tvm {
namespace relay {
namespace pass {

using tvm::IRPrinter;

class ModulePass;

/*!
 * \brief Module-level passes are designed to implement global
 * analysis/optimizations, i.e. interprocedural optimizations (IPO), etc. Passes
 * at this level have the full control of a given Relay program including
 * addition and deletion of functions.
 */
class ModulePassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The pass function sketches the real optimization. For example,
   * we may need to perform dead code elimination on the module level. We could
   * implement the algorithm in the `pass_func` and let it run on a module. It
   * will then remove the dead code including the unused functions in the module.
   */
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func;

  ModulePassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("pass_info", &pass_info);
  }

  /*!
   * \brief Run a module pass on a certain module.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return Return the updated module.
   */
  Module operator()(const Module& mod) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const { return pass_info; }

  /*!
   * \brief Set the context information for a module pass.
   *
   * \param pass_ctx The context information for a module pass.
   */
  void SetContext(const PassContext& pass_ctx) final;

  TVM_DLL static ModulePass make(
      runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func,
      PassInfo pass_info);

  static constexpr const char* _type_key = "relay.ModulePass";
  TVM_DECLARE_NODE_TYPE_INFO(ModulePassNode, PassNode);

 private:
  /*!
   * \brief The context information that is used to help perform a module pass.
   */
  PassContext pass_ctx_;
};

RELAY_DEFINE_NODE_REF(ModulePass, ModulePassNode, Pass);

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
  runtime::TypedPackedFunc<Function(Function, PassContext)> pass_func;

  FunctionPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("pass_info", &pass_info);
  }

  /*!
   * \brief Run a function pass on a certain module.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return Return the updated module.
   */
  Module operator()(const Module& mod) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const { return pass_info; }

  /*!
   * \brief Set the context information for a function-level pass.
   *
   * \param pass_ctx The context information for a function-level pass.
   */
  void SetContext(const PassContext& pass_ctx) final;

  TVM_DLL static FunctionPass make(
      runtime::TypedPackedFunc<Function(Function, PassContext)> pass_func,
      PassInfo pass_info);

  static constexpr const char* _type_key = "relay.FunctionPass";
  TVM_DECLARE_NODE_TYPE_INFO(FunctionPassNode, PassNode);

 private:
  /*
   * \brief Check if a function should be skipped for optimization.
   *
   * \param func The target function to be checked.
   *
   * \return Return true if the function will be skipped, otherwise false.
   */
  bool SkipFunction(const Function& func) const;

  /*!
   * \brief The context information that is used to help perform a module pass.
   */
  PassContext pass_ctx_;
};

RELAY_DEFINE_NODE_REF(FunctionPass, FunctionPassNode, Pass);

class SequentialPass;

/*!
 * \brief The SequentialPassNode contains a set of passes that transform Relay
 * programs from one AST to another semantically equivalent one.
 *
 * One example of this level of pass is that the pass manager needs to correctly
 * perform a host of optimizations with a given optimization level and disabled
 * passes.
 */
class SequentialPassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief A list of passes that used to compose a sequential pass. */
  tvm::Array<Pass> passes;
  /*!
   * \brief A list of disabled passes that should be excluded when executing the
   * sequential pass.
   */
  tvm::Array<tvm::Expr> disabled;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("pass_info", &pass_info);
    v->Visit("passes", &passes);
    v->Visit("disabled", &disabled);
  }

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const { return pass_info; }

  /*!
   * \brief Add a pass to the pass list.
   *
   * \param pass The candidate pass to be added.
   */
  void AddPass(const Pass& pass) {
    passes.push_back(pass);
  }

  TVM_DLL static SequentialPass make(tvm::Array<Pass> passes,
                                     PassInfo pass_info,
                                     tvm::Array<tvm::Expr> disabled);

  /*!
   * \brief Resolve the pass dependency. It globs all required passes by
   *        a given pass and executes them.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return The updated module after resolving pass dependencies.
   *
   * TODO(zhiics) Build a dependency graph among the passes using provided
   * metadata, i.e. required_passes. Likely, we can have a data structure, i.e.
   * PassInfo, to store the relevant information including the parent passes.
   */
  void ResolveDependency(const Module& mod);

  TVM_DLL std::vector<std::string> DisabledPasses() const;

  /*!
   * \brief Perform optimizations on a series of passes. The aforementioned
   *        typical pass manager jobs could be done by it. This function could
   *        be overloaded to focus on different metrics, i.e. performance,
   *        memory footprint, etc.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return Return the updated module.
   */
  Module operator()(const Module& mod) const final;

  /*!
   * \brief Set the context information for a sequential pass.
   *
   * \param pass_ctx The context information for a sequential pass.
   */
  void SetContext(const PassContext& pass_ctx) final;

  static constexpr const char* _type_key = "relay.SequentialPass";
  TVM_DECLARE_NODE_TYPE_INFO(SequentialPassNode, PassNode);

 private:
  /*!
   * \brief The context information that is used to help perform a module pass.
   */
  PassContext pass_ctx_;
};

RELAY_DEFINE_NODE_REF(SequentialPass, SequentialPassNode, Pass);

PassInfo PassInfoNode::make(int opt_level, std::string name,
                            tvm::Array<tvm::Expr> required) {
  auto pass_info = make_node<PassInfoNode>();
  pass_info->opt_level = opt_level;
  pass_info->name = std::move(name);
  pass_info->required = std::move(required);
  return PassInfo(pass_info);
}

PassContext PassContextNode::make() {
  auto ctx = make_node<PassContextNode>();
  return PassContext(ctx);
}

ModulePass ModulePassNode::make(
    runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func,
    PassInfo pass_info) {
  auto n = make_node<ModulePassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  return ModulePass(n);
}

// Module -> Module optimizations.
// TODO(zhiics) 1. Check and handle the required passes.
//              2. Probably use CoW for all places that use module instead of
//              returning the updated one.
Module ModulePassNode::operator()(const Module& mod) const {
  PassInfo pass_info = Info();
  LOG(INFO) << "Executing module pass : " << pass_info.operator->()->name
            << " with opt level: " << pass_info.operator->()->opt_level << "\n";
  CHECK(mod.defined());
  auto updated_mod = pass_func(mod, pass_ctx_);
  CHECK(updated_mod.defined());
  return updated_mod;
}

void ModulePassNode::SetContext(const PassContext& pass_ctx) {
  pass_ctx_ = pass_ctx;
}

FunctionPass FunctionPassNode::make(
    runtime::TypedPackedFunc<Function(Function, PassContext)> pass_func,
    PassInfo pass_info) {
  auto n = make_node<FunctionPassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  return FunctionPass(n);
}

// Perform Module -> Module optimizations at the Function level.
// TODO(zhiics) Check and handle the required passes.
Module FunctionPassNode::operator()(const Module& mod) const {
  PassInfo pass_info = Info();
  LOG(INFO) << "Executing function pass : " << pass_info.operator->()->name
            << " with opt level: " << pass_info.operator->()->opt_level << "\n";
  CHECK(mod.defined());
  std::vector<std::pair<GlobalVar, Function>> updated_funcs;
  ModuleNode* mod_node = mod.operator->();
  for (const auto& it : mod_node->functions) {
    if (!SkipFunction(it.second)) {
      auto updated_func = pass_func(it.second, pass_ctx_);
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

void FunctionPassNode::SetContext(const PassContext& pass_ctx) {
  pass_ctx_ = pass_ctx;
}

// TODO(zhiics) Create an enum attribute for FunctionNode
// enum Attribute {kPrimitive, kSkipOptimization}
bool FunctionPassNode::SkipFunction(const Function& func) const {
  NodeRef res = FunctionGetAttr(func, "SkipOptimization");
  const ir::IntImm* pval = res.as<ir::IntImm>();
  return pval && pval->value != 0;
}

SequentialPass SequentialPassNode::make(tvm::Array<Pass> passes,
                                        PassInfo pass_info,
                                        tvm::Array<tvm::Expr> disabled) {
  auto n = make_node<SequentialPassNode>();
  n->passes = std::move(passes);
  n->pass_info = std::move(pass_info);
  n->disabled = std::move(disabled);
  return SequentialPass(n);
}

// TODO(jroesch, zhiics): we currenlty only sequentially execute each pass in
// a SequentialPass without the consideration of their orders. The phase
// ordering problem needed to be handled in the future.
Module SequentialPassNode::operator()(const Module& module) const {
  Module mod = module;
  for (const Pass& pass : passes) {
    CHECK(pass.defined()) << "Found undefined pass for optimization.";
    const auto* pn = pass.operator->();
    mod = (*pn)(mod);
  }
  return mod;
}

void SequentialPassNode::ResolveDependency(const Module& mod) {
  // TODO(zhiics) Implement it.
  // 1. Consider the required passes for each pass.
  // 2. Only resolve the enabled passes.
  // 3. Build a dependency graph. Probably we need to update the pass list.
  LOG(FATAL) << "Pass dependency has not been resolved yet."
             << "\n";
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

void SequentialPassNode::SetContext(const PassContext& pass_ctx) {
  pass_ctx_ = pass_ctx;
}

Pass CreateModulePass(
    const runtime::TypedPackedFunc<Module(Module, PassContext)>& pass_func,
    int opt_level,
    const std::string& name,
    const tvm::Array<tvm::Expr>& required) {
  PassInfo pass_info = PassInfoNode::make(opt_level, name, required);
  return ModulePassNode::make(pass_func, pass_info);
}

Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, PassContext)>& pass_func,
    int opt_level,
    const std::string& name,
    const tvm::Array<tvm::Expr>& required) {
  PassInfo pass_info = PassInfoNode::make(opt_level, name, required);
  return FunctionPassNode::make(pass_func, pass_info);
}

Pass CreateSequentialPass(const tvm::Array<Pass>& passes,
                          int opt_level,
                          const std::string& name,
                          const tvm::Array<tvm::Expr>& required,
                          const tvm::Array<tvm::Expr>& disabled) {
  PassInfo pass_info = PassInfoNode::make(opt_level, name, required);
  return SequentialPassNode::make(passes, pass_info, disabled);
}

TVM_REGISTER_NODE_TYPE(PassInfoNode);

TVM_REGISTER_API("relay._ir_pass.PassInfo")
.set_body_typed(PassInfoNode::make);

TVM_REGISTER_API("relay._ir_pass.Info")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Pass pass = args[0];
  *ret = pass->Info();
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PassInfoNode>([](const PassInfoNode* node,
                                tvm::IRPrinter* p) {
  p->stream << "The meta data of the pass: ";
  p->stream << "pass name: " << node->name;
  p->stream << "opt_level: " << node->opt_level;
  p->stream << "required passes: [" << "\n";
  for (const auto& it : node->required) {
    const auto* str = it.as<tvm::ir::StringImm>();
    p->stream << str->value << ", ";
  }
  p->stream << "]\n";
});

TVM_REGISTER_NODE_TYPE(ModulePassNode);

TVM_REGISTER_API("relay._ir_pass.CreateModulePass")
.set_body_typed(CreateModulePass);

TVM_REGISTER_API("relay._ir_pass.RunPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Pass pass = args[0];
  Module mod = args[1];
  CHECK(pass.defined())
      << "Running an undefined pass is not allowed."
      << "\n";

  const auto* pn = pass.operator->();
  *ret = (*pn)(mod);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ModulePassNode>([](const ModulePassNode* node,
                                 tvm::IRPrinter* p) {
  const PassInfoNode* pn = node->Info().operator->();
  p->stream << "Run Module pass: " << pn->name
            << " at the optimization level " << pn->opt_level;
});

TVM_REGISTER_NODE_TYPE(FunctionPassNode);

TVM_REGISTER_API("relay._ir_pass.CreateFunctionPass")
.set_body_typed(CreateFunctionPass);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<FunctionPassNode>([](const FunctionPassNode* node,
                                   tvm::IRPrinter* p) {
  const PassInfoNode* pn = node->Info().operator->();
  p->stream << "Run Function pass: " << pn->name
            << " at the optimization level " << pn->opt_level;
});

TVM_REGISTER_NODE_TYPE(SequentialPassNode);

TVM_REGISTER_API("relay._ir_pass.CreateSequentialPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  tvm::Array<Pass> passes = args[0];
  int opt_level = args[1];
  std::string name = args[2];
  tvm::Array<tvm::Expr> required = args[3];
  tvm::Array<tvm::Expr> disabled = args[4];
  PassInfo pass_info = PassInfoNode::make(opt_level, name, required);
  *ret = SequentialPassNode::make(passes, pass_info, disabled);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<SequentialPassNode>([](const SequentialPassNode* node,
                                     tvm::IRPrinter* p) {
  const PassInfoNode* seq_pn = node->Info().operator->();
  p->stream << "Run SequentialPass pass: " << seq_pn->name
            << " at the optimization level. " << seq_pn->opt_level;
  p->stream << "The passes will be executed are: [";
  for (const auto& it : node->passes) {
    const PassNode* pn = it.operator->();
    const PassInfoNode* pass_info_node = pn->Info().operator->();
    p->stream << pass_info_node->name << " ";
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
.set_body_typed(PassContextNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PassContextNode>([](const PassContextNode* node,
                                tvm::IRPrinter* p) {
    p->stream << "TODO(zhiics): printing context";
    LOG(FATAL) << "PassContext printer has not been implemented yet."
               << "\n";
});

}  // namespace pass
}  // namespace relay
}  // namespace tvm

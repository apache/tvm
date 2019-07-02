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
#include <dmlc/thread_local.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/device_api.h>

#include <algorithm>
#include <stack>
#include <unordered_set>

namespace tvm {
namespace relay {
namespace transform {

using tvm::IRPrinter;

struct RelayPassContextThreadLocalEntry {
  /*! \brief The default pass context. */
  PassContext default_context;

  /*! \brief The current pass context. */
  std::stack<PassContext> context_stack;

  RelayPassContextThreadLocalEntry() {
    default_context = PassContext(make_node<PassContextNode>());
  }
};

/*! \brief Thread local store to hold the pass context. */
typedef dmlc::ThreadLocalStore<RelayPassContextThreadLocalEntry>
    RelayPassContextThreadLocalStore;

void PassContext::EnterWithScope() {
  RelayPassContextThreadLocalEntry* entry =
      RelayPassContextThreadLocalStore::Get();
  entry->context_stack.push(*this);
}

void PassContext::ExitWithScope() {
  RelayPassContextThreadLocalEntry* entry =
      RelayPassContextThreadLocalStore::Get();
  CHECK(!entry->context_stack.empty());
  CHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();
}

PassContext PassContext::Current() {
  RelayPassContextThreadLocalEntry* entry =
      RelayPassContextThreadLocalStore::Get();
  if (!entry->context_stack.empty()) {
    return entry->context_stack.top();
  } else {
    return entry->default_context;
  }
}

PassContext PassContext::Create() {
  return PassContext(make_node<PassContextNode>());
}

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
   * \brief Run a module pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param mod The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const { return pass_info; }

  TVM_DLL static ModulePass make(
      runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func,
      PassInfo pass_info);

  static constexpr const char* _type_key = "relay.ModulePass";
  TVM_DECLARE_NODE_TYPE_INFO(ModulePassNode, PassNode);
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
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func;

  FunctionPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
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
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const { return pass_info; }

  TVM_DLL static FunctionPass make(
      runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func,
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
};

RELAY_DEFINE_NODE_REF(FunctionPass, FunctionPassNode, Pass);

/*!
 * \brief The SequentialNode contains a set of passes that transform Relay
 * programs from one AST to another semantically equivalent one.
 *
 * One example of this level of pass is that the pass manager needs to correctly
 * perform a host of optimizations with a given optimization level and disabled
 * passes.
 */
class SequentialNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief A list of passes that used to compose a sequential pass. */
  tvm::Array<Pass> passes;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("pass_info", &pass_info);
    v->Visit("passes", &passes);
  }

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const { return pass_info; }

  /*!
   * \brief Check if a pass is enabled.
   *
   * \param info The pass information.
   *
   * \return true if the pass is enabled. Otherwise, false.
   */
  bool PassEnabled(const PassInfo& info) const;

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

  /*!
   * \brief Perform optimizations on a series of passes. The aforementioned
   *        typical pass manager jobs could be done by it. This function could
   *        be overloaded to focus on different metrics, i.e. performance,
   *        memory footprint, etc.
   *
   * \param mod The module that these passes are applied on.
   * \param pass_ctx The context that these passes execute on.
   *
   * \return Return the updated module.
   */
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;

  static constexpr const char* _type_key = "relay.Sequential";
  TVM_DECLARE_NODE_TYPE_INFO(SequentialNode, PassNode);
};

PassInfo PassInfoNode::make(int opt_level,
                            std::string name,
                            tvm::Array<tvm::Expr> required) {
  auto pass_info = make_node<PassInfoNode>();
  pass_info->opt_level = opt_level;
  pass_info->name = std::move(name);
  pass_info->required = std::move(required);
  return PassInfo(pass_info);
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
Module ModulePassNode::operator()(const Module& mod,
                                  const PassContext& pass_ctx) const {
  const PassInfo& pass_info = Info();
  DLOG(INFO) << "Executing module pass : "
             << pass_info->name
             << " with opt level: "
             << pass_info->opt_level;
  CHECK(mod.defined());
  Module updated_mod = pass_func(mod, pass_ctx);
  CHECK(updated_mod.defined());
  return updated_mod;
}

FunctionPass FunctionPassNode::make(
    runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func,
    PassInfo pass_info) {
  auto n = make_node<FunctionPassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  return FunctionPass(n);
}

// Perform Module -> Module optimizations at the Function level.
Module FunctionPassNode::operator()(const Module& mod,
                                    const PassContext& pass_ctx) const {
  const PassInfo& pass_info = Info();
  CHECK(mod.defined());
  DLOG(INFO) << "Executing function pass : "
             << pass_info->name
             << " with opt level: "
             << pass_info->opt_level;

  Module updated_mod = mod;
  // Execute the pass function and return a new module.
  std::vector<std::pair<GlobalVar, Function> > updates;
  auto original = mod->functions;
  for (const auto& it : original) {
    auto updated_func = SkipFunction(it.second)
                            ? it.second
                            : pass_func(it.second, updated_mod, pass_ctx);
    updates.push_back({it.first, updated_func});
  }

  for (const auto& pair : updates) {
    updated_mod->Add(pair.first, pair.second, true);
  }
  return updated_mod;
}

// TODO(zhiics) Create an enum attribute for FunctionNode
// enum Attribute {kPrimitive, kSkipOptimization}
bool FunctionPassNode::SkipFunction(const Function& func) const {
  NodeRef res = FunctionGetAttr(func, "SkipOptimization");
  const ir::IntImm* pval = res.as<ir::IntImm>();
  return pval && pval->value != 0;
}

Sequential::Sequential(tvm::Array<Pass> passes, PassInfo pass_info) {
  auto n = make_node<SequentialNode>();
  n->passes = std::move(passes);
  n->pass_info = std::move(pass_info);
  node_ = std::move(n);
}

Sequential::Sequential(tvm::Array<Pass> passes, std::string name) {
  auto n = make_node<SequentialNode>();
  n->passes = std::move(passes);
  PassInfo pass_info = PassInfoNode::make(2, std::move(name), {});
  n->pass_info = std::move(pass_info);
  node_ = std::move(n);
}

const SequentialNode* Sequential::operator->() const {
  return static_cast<const SequentialNode*>(this->node_.get());
}

void SequentialNode::ResolveDependency(const Module& mod) {
  // TODO(zhiics) Implement it.
  // 1. Consider the required passes for each pass.
  // 2. Only resolve the enabled passes.
  // 3. Build a dependency graph. Probably we need to update the pass list.
  LOG(FATAL) << "Pass dependency has not been resolved yet."
             << "\n";
}

// linearly scan the pass array to match pass_name
inline bool PassArrayContains(const Array<tvm::Expr>& pass_array,
                              const std::string& pass_name) {
  for (auto x : pass_array) {
    auto* str_name = x.as<ir::StringImm>();
    CHECK(str_name) << "pass name must be str";
    if (str_name->value == pass_name) return true;
  }
  return false;
}

bool SequentialNode::PassEnabled(const PassInfo& info) const {
  PassContext ctx = PassContext::Current();

  if (PassArrayContains(ctx->disabled_pass, info->name)) {
    return false;
  }

  if (PassArrayContains(ctx->required_pass, info->name)) {
    return true;
  }

  return ctx->opt_level >= info->opt_level;
}

Pass GetPass(const std::string& pass_name) {
  using tvm::runtime::Registry;
  std::string fpass_name = "relay._transform." + pass_name;
  const auto* f = Registry::Get(fpass_name);
  CHECK(f != nullptr) << "Cannot find " << fpass_name
                      << "to create the pass " << pass_name;
  return (*f)();
}

// TODO(zhiics): we currenlty only sequentially execute each pass in
// a Sequential without the consideration of their orders. The phase
// ordering problem needs to be handled in the future.
Module SequentialNode::operator()(const Module& module,
                                  const PassContext& pass_ctx) const {
  Module mod = module;
  for (const Pass& pass : passes) {
    CHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!PassEnabled(pass_info))  continue;
    // resolve dependencies
    for (const auto& it : pass_info->required) {
      const auto* name = it.as<tvm::ir::StringImm>();
      CHECK(name);
      mod = GetPass(name->value)(mod, pass_ctx);
    }
    mod = pass(mod, pass_ctx);
  }
  return mod;
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
    const runtime::TypedPackedFunc<Function(Function, Module, PassContext)>& pass_func,
    int opt_level,
    const std::string& name,
    const tvm::Array<tvm::Expr>& required) {
  PassInfo pass_info = PassInfoNode::make(opt_level, name, required);
  return FunctionPassNode::make(pass_func, pass_info);
}

TVM_REGISTER_NODE_TYPE(PassInfoNode);

TVM_REGISTER_API("relay._transform.PassInfo")
.set_body_typed(PassInfoNode::make);

TVM_REGISTER_API("relay._transform.Info")
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

TVM_REGISTER_API("relay._transform.MakeModulePass")
.set_body_typed(ModulePassNode::make);

TVM_REGISTER_API("relay._transform.RunPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Pass pass = args[0];
  Module mod = args[1];
  *ret = pass(mod);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ModulePassNode>([](const ModulePassNode* node,
                                 tvm::IRPrinter* p) {
  const PassInfo info = node->Info();
  p->stream << "Run Module pass: " << info->name
            << " at the optimization level " << info->opt_level;
});

TVM_REGISTER_NODE_TYPE(FunctionPassNode);

TVM_REGISTER_API("relay._transform.MakeFunctionPass")
.set_body_typed(FunctionPassNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<FunctionPassNode>([](const FunctionPassNode* node,
                                   tvm::IRPrinter* p) {
  const PassInfo info = node->Info();
  p->stream << "Run Function pass: " << info->name
            << " at the optimization level " << info->opt_level;
});

TVM_REGISTER_NODE_TYPE(SequentialNode);

TVM_REGISTER_API("relay._transform.Sequential")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  tvm::Array<Pass> passes = args[0];
  int opt_level = args[1];
  std::string name = args[2];
  tvm::Array<tvm::Expr> required = args[3];
  PassInfo pass_info = PassInfoNode::make(opt_level, name, required);
  *ret = Sequential(passes, pass_info);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<SequentialNode>([](const SequentialNode* node,
                                 tvm::IRPrinter* p) {
  const PassInfo info = node->Info();
  p->stream << "Run Sequential pass: " << info->name
            << " at the optimization level " << info->opt_level << ". ";
  p->stream << "The passes will be executed are: [";
  for (const auto& it : node->passes) {
    const PassInfo pass_info = it->Info();
    p->stream << pass_info->name << " ";
  }
  p->stream << "]";
});

TVM_REGISTER_NODE_TYPE(PassContextNode);

TVM_REGISTER_API("relay._transform.PassContext")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  auto pctx = PassContext::Create();
  int opt_level = args[0];
  int fallback_device = args[1];
  tvm::Array<tvm::Expr> required = args[2];
  tvm::Array<tvm::Expr> disabled = args[3];
  pctx->opt_level = opt_level;
  pctx->fallback_device = fallback_device;
  pctx->required_pass = std::move(required);
  pctx->disabled_pass = std::move(disabled);
  *ret = pctx;
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PassContextNode>([](const PassContextNode* node,
                               tvm::IRPrinter* p) {
  p->stream << "Pass context information: " << "\n";
  p->stream << "\topt_level: " << node->opt_level << "\n";
  p->stream << "\tfallback device: "
            << runtime::DeviceName(node->fallback_device)
            << "\n";

  p->stream << "\trequired passes: [" << node->opt_level;
  for (const auto& it : node->required_pass) {
    p->stream << it << " ";
  }
  p->stream << "]\n";

  p->stream << "\tdisabled passes: [" << node->opt_level;
  for (const auto& it : node->disabled_pass) {
    p->stream << it << " ";
  }
  p->stream << "]";
});

class PassContext::Internal {
 public:
  static void EnterScope(PassContext pass_ctx) {
    pass_ctx.EnterWithScope();
  }

  static void ExitScope(PassContext pass_ctx) {
    pass_ctx.ExitWithScope();
  }
};

TVM_REGISTER_API("relay._transform.GetCurrentPassContext")
.set_body_typed(PassContext::Current);

TVM_REGISTER_API("relay._transform.EnterPassContext")
.set_body_typed(PassContext::Internal::EnterScope);

TVM_REGISTER_API("relay._transform.ExitPassContext")
.set_body_typed(PassContext::Internal::ExitScope);

}  // namespace transform
}  // namespace relay
}  // namespace tvm

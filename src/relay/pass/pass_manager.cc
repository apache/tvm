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

/*!
 * \brief A data structure to map the names of specific optimizations to
 *        numeric optimization levels
 */
class OptPassLevel {
 public:
  /*!
   * \brief Get level for an optimization pass
   *
   * \param key pass name
   * \return int level
   */
  int operator[](const std::string& key) const {
    const auto data = CreateMap();
    auto it = data.find(key);
    if (it == data.end()) {
      return -1;
    }
    return it->second;
  }

 private:
  static const std::unordered_map<std::string, int> CreateMap() {
    const std::unordered_map<std::string, int> m = {
      {"SimplifyInference", 0},
      {"OpFusion", 1},
      {"FoldConstant", 2},
      {"CombineParallelConv2D", 3},
      {"FoldScaleAxis", 3},
      {"AlterOpLayout", 3},
      {"CanonicalizeOps", 3},
      {"EliminateCommonSubexpr", 3}
    };
    return m;
  }
};

PassContext::PassContext(int opt_level, int fallback_device,
                         tvm::Array<tvm::Expr> required_pass,
                         tvm::Array<tvm::Expr> disabled_pass) {
  auto ctx = make_node<PassContextNode>();
  ctx->opt_level = opt_level;
  ctx->fallback_device = fallback_device;
  ctx->required_pass = std::move(required_pass);
  ctx->disabled_pass = std::move(disabled_pass);
  node_ = std::move(ctx);
}

const PassContextNode* PassContext::operator->() const {
  return static_cast<const PassContextNode*>(node_.get());
}

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

  /*!
   * \brief A helper struct to get the optimization pass name to opt level
   * mapping.
   */
  OptPassLevel opt_pass_level;

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
   * \brief Add a pass to the pass list.
   *
   * \param pass The candidate pass to be added.
   */
  void AddPass(const Pass& pass) {
    passes.push_back(pass);
  }

  /*!
   * \brief Check if a pass is enabled.
   *
   * \param pass_name The name of an optimization/analysis pass.
   *
   * \return true if the pass is enabled. Otherwise, false.
   */
  bool pass_enabled(const std::string& pass_name) const;

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

  std::unordered_set<std::string> DisabledPasses(
      const Array<tvm::Expr>& disabled) const;

  std::unordered_set<std::string> RequiredPasses(
      const Array<tvm::Expr>& disabled) const;

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

PassInfo PassInfoNode::make(int opt_level, std::string name,
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
// TODO(zhiics) Check and handle the required passes.
Module ModulePassNode::operator()(const Module& mod,
                                  const PassContext& pass_ctx) const {
  PassInfo pass_info = Info();
  LOG(INFO) << "Executing module pass : " << pass_info.operator->()->name
            << " with opt level: " << pass_info.operator->()->opt_level << "\n";

  CHECK(mod.defined());
  auto updated_mod = pass_func(mod, pass_ctx);
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
// TODO(zhiics) Check and handle the required passes.
Module FunctionPassNode::operator()(const Module& mod,
                                    const PassContext& pass_ctx) const {
  PassInfo pass_info = Info();
  LOG(INFO) << "Executing function pass : " << pass_info.operator->()->name
            << " with opt level: " << pass_info.operator->()->opt_level << "\n";
  CHECK(mod.defined());
  Module new_mod = ModuleNode::make({}, mod->type_definitions);

  // Execute the pass function and return a new module.
  for (const auto& it : mod->functions) {
    auto updated_func = SkipFunction(it.second) ? it.second : pass_func(it.second, mod, pass_ctx);
    new_mod->Add(it.first, updated_func);
  }

  return new_mod;
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

std::unordered_set<std::string> SequentialNode::DisabledPasses(
    const Array<tvm::Expr>& disabled) const {
  std::unordered_set<std::string> ret;
  for (const auto& it : disabled) {
    const auto* str = it.as<tvm::ir::StringImm>();
    CHECK(str) << "disabled passes must be string.";
    ret.emplace(str->value);
  }
  return ret;
}

std::unordered_set<std::string> SequentialNode::RequiredPasses(
    const Array<tvm::Expr>& required) const {
  std::unordered_set<std::string> ret;
  for (const auto& it : required) {
    const auto* str = it.as<tvm::ir::StringImm>();
    CHECK(str) << "disabled passes must be string.";
    ret.emplace(str->value);
  }
  return ret;
}

bool SequentialNode::pass_enabled(const std::string& pass_name) const {
  PassContext ctx = PassContext::Current();

  const PassContextNode* ctx_node = ctx.operator->();
  auto required = RequiredPasses(ctx_node->required_pass);
  auto disabled = DisabledPasses(ctx_node->required_pass);

  if (disabled.count(pass_name)) {
    return false;
  }

  if (required.count(pass_name)) {
    return true;
  }
  return ctx_node->opt_level >= opt_pass_level[pass_name];
}

// TODO(zhiics): we currenlty only sequentially execute each pass in
// a Sequential without the consideration of their orders. The phase
// ordering problem needed to be handled in the future.
Module SequentialNode::operator()(const Module& module,
                                  const PassContext& pass_ctx) const {
  const auto* ctx_node = pass_ctx.operator->();
  int opt_level = ctx_node->opt_level;
  auto disabled = DisabledPasses(ctx_node->disabled_pass);
  Module mod = module;
  for (const Pass& pass : passes) {
    CHECK(pass.defined()) << "Found undefined pass for optimization.";
    PassInfo info = pass->Info();
    const auto& pass_name = info.operator->()->name;
    const auto& pass_opt_level = info.operator->()->opt_level;
    // Skip the pass if its optimization level is higher that the  one of in the
    // pass context or if this pass is disabled.
    if (pass_opt_level > opt_level || disabled.count(pass_name)) {
      continue;
    }
    const auto* pn = pass.operator->();
    mod = (*pn)(mod, pass_ctx);
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

TVM_REGISTER_API("relay._transform.CreateModulePass")
.set_body_typed(CreateModulePass);

TVM_REGISTER_API("relay._transform.RunPass")
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

TVM_REGISTER_API("relay._transform.CreateFunctionPass")
.set_body_typed(CreateFunctionPass);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<FunctionPassNode>([](const FunctionPassNode* node,
                                   tvm::IRPrinter* p) {
  const PassInfoNode* pn = node->Info().operator->();
  p->stream << "Run Function pass: " << pn->name
            << " at the optimization level " << pn->opt_level;
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
  const PassInfoNode* seq_pn = node->Info().operator->();
  p->stream << "Run Sequential pass: " << seq_pn->name
            << " at the optimization level " << seq_pn->opt_level << ". ";
  p->stream << "The passes will be executed are: [";
  for (const auto& it : node->passes) {
    const PassNode* pn = it.operator->();
    const PassInfoNode* pass_info_node = pn->Info().operator->();
    p->stream << pass_info_node->name << " ";
  }
  p->stream << "]";
});

TVM_REGISTER_NODE_TYPE(PassContextNode);

TVM_REGISTER_API("relay._transform.PassContext")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  int opt_level = args[0];
  int fallback_device = args[1];
  tvm::Array<tvm::Expr> required = args[2];
  tvm::Array<tvm::Expr> disabled = args[3];
  *ret = PassContext(opt_level, fallback_device, required, disabled);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PassContextNode>([](const PassContextNode* node,
                               tvm::IRPrinter* p) {
  p->stream << "Pass context information: " << "\n";
  p->stream << "\topt_level: " << node->opt_level << "\n";
  p->stream << "\tfallback device: " << runtime::DeviceName(node->opt_level)
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

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
 * \file src/ir/transform.cc
 * \brief Infrastructure for transformation passes.
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/node/repr_printer.h>
#include <tvm/ir/transform.h>

// TODO(tqchen): Update to use String container after it is merged.
#include <tvm/tir/expr.h>

#include <stack>
#include <unordered_set>

namespace tvm {
namespace transform {

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;
using tvm::ReprPrinter;

struct PassContextThreadLocalEntry {
  /*! \brief The default pass context. */
  PassContext default_context;

  /*! \brief The current pass context. */
  std::stack<PassContext> context_stack;

  PassContextThreadLocalEntry() {
    default_context = PassContext(make_object<PassContextNode>());
  }
};

/*! \brief Thread local store to hold the pass context. */
typedef dmlc::ThreadLocalStore<PassContextThreadLocalEntry>
    RelayPassContextThreadLocalStore;

void PassContext::EnterWithScope() {
  PassContextThreadLocalEntry* entry =
      RelayPassContextThreadLocalStore::Get();
  entry->context_stack.push(*this);
}

void PassContext::ExitWithScope() {
  PassContextThreadLocalEntry* entry =
      RelayPassContextThreadLocalStore::Get();
  CHECK(!entry->context_stack.empty());
  CHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();
}

PassContext PassContext::Current() {
  PassContextThreadLocalEntry* entry =
      RelayPassContextThreadLocalStore::Get();
  if (!entry->context_stack.empty()) {
    return entry->context_stack.top();
  } else {
    return entry->default_context;
  }
}

PassContext PassContext::Create() {
  return PassContext(make_object<PassContextNode>());
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
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func;

  ModulePassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
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
  IRModule operator()(const IRModule& mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  static constexpr const char* _type_key = "relay.ModulePass";
  TVM_DECLARE_FINAL_OBJECT_INFO(ModulePassNode, PassNode);
};

class ModulePass : public Pass {
 public:
  ModulePass(runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func,
             PassInfo pass_info);

  TVM_DEFINE_OBJECT_REF_METHODS(ModulePass, Pass, ModulePassNode);
};

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

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pass_info", &pass_info);
    v->Visit("passes", &passes);
  }

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

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
  void ResolveDependency(const IRModule& mod);

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
  IRModule operator()(const IRModule& mod, const PassContext& pass_ctx) const final;

  static constexpr const char* _type_key = "relay.Sequential";
  TVM_DECLARE_FINAL_OBJECT_INFO(SequentialNode, PassNode);
};

PassInfo::PassInfo(int opt_level,
                   std::string name,
                   tvm::Array<tvm::PrimExpr> required) {
  auto pass_info = make_object<PassInfoNode>();
  pass_info->opt_level = opt_level;
  pass_info->name = std::move(name);
  pass_info->required = std::move(required);
  data_ = std::move(pass_info);
}

ModulePass::ModulePass(
    runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func,
    PassInfo pass_info) {
  auto n = make_object<ModulePassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

// Module -> Module optimizations.
IRModule ModulePassNode::operator()(const IRModule& mod,
                                    const PassContext& pass_ctx) const {
  const PassInfo& pass_info = Info();
  DLOG(INFO) << "Executing module pass : "
             << pass_info->name
             << " with opt level: "
             << pass_info->opt_level;
  CHECK(mod.defined());
  IRModule updated_mod = pass_func(mod, pass_ctx);
  CHECK(updated_mod.defined());
  return updated_mod;
}

Sequential::Sequential(tvm::Array<Pass> passes, PassInfo pass_info) {
  auto n = make_object<SequentialNode>();
  n->passes = std::move(passes);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

Sequential::Sequential(tvm::Array<Pass> passes, std::string name) {
  auto n = make_object<SequentialNode>();
  n->passes = std::move(passes);
  PassInfo pass_info = PassInfo(2, std::move(name), {});
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

const SequentialNode* Sequential::operator->() const {
  return static_cast<const SequentialNode*>(get());
}

void SequentialNode::ResolveDependency(const IRModule& mod) {
  // TODO(zhiics) Implement it.
  // 1. Consider the required passes for each pass.
  // 2. Only resolve the enabled passes.
  // 3. Build a dependency graph. Probably we need to update the pass list.
  LOG(FATAL) << "Pass dependency has not been resolved yet."
             << "\n";
}

// linearly scan the pass array to match pass_name
inline bool PassArrayContains(const Array<tvm::PrimExpr>& pass_array,
                              const std::string& pass_name) {
  for (auto x : pass_array) {
    auto* str_name = x.as<tir::StringImmNode>();
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
IRModule SequentialNode::operator()(const IRModule& module,
                                  const PassContext& pass_ctx) const {
  IRModule mod = module;
  for (const Pass& pass : passes) {
    CHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!PassEnabled(pass_info))  continue;
    // resolve dependencies
    for (const auto& it : pass_info->required) {
      const auto* name = it.as<tvm::tir::StringImmNode>();
      CHECK(name);
      mod = GetPass(name->value)(mod, pass_ctx);
    }
    mod = pass(mod, pass_ctx);
  }
  return mod;
}

Pass CreateModulePass(
    const runtime::TypedPackedFunc<IRModule(IRModule, PassContext)>& pass_func,
    int opt_level,
    const std::string& name,
    const tvm::Array<tvm::PrimExpr>& required) {
  PassInfo pass_info = PassInfo(opt_level, name, required);
  return ModulePass(pass_func, pass_info);
}

TVM_REGISTER_NODE_TYPE(PassInfoNode);

TVM_REGISTER_GLOBAL("relay._transform.PassInfo")
.set_body_typed([](int opt_level, std::string name, tvm::Array<PrimExpr> required) {
  return PassInfo(opt_level, name, required);
});

TVM_REGISTER_GLOBAL("relay._transform.Info")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Pass pass = args[0];
  *ret = pass->Info();
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<PassInfoNode>([](const ObjectRef& ref, tvm::ReprPrinter* p) {
  auto* node = static_cast<const PassInfoNode*>(ref.get());
  p->stream << "The meta data of the pass: ";
  p->stream << "pass name: " << node->name;
  p->stream << "opt_level: " << node->opt_level;
  p->stream << "required passes: [" << "\n";
  for (const auto& it : node->required) {
    const auto* str = it.as<tvm::tir::StringImmNode>();
    p->stream << str->value << ", ";
  }
  p->stream << "]\n";
});

TVM_REGISTER_NODE_TYPE(ModulePassNode);

TVM_REGISTER_GLOBAL("relay._transform.MakeModulePass")
.set_body_typed(
  [](runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func,
     PassInfo pass_info) {
  return ModulePass(pass_func, pass_info);
});

TVM_REGISTER_GLOBAL("relay._transform.RunPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Pass pass = args[0];
  IRModule mod = args[1];
  *ret = pass(mod);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ModulePassNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const ModulePassNode*>(ref.get());
  const PassInfo info = node->Info();
  p->stream << "Run Module pass: " << info->name
            << " at the optimization level " << info->opt_level;
});

TVM_REGISTER_NODE_TYPE(SequentialNode);

TVM_REGISTER_GLOBAL("relay._transform.Sequential")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  tvm::Array<Pass> passes = args[0];
  int opt_level = args[1];
  std::string name = args[2];
  tvm::Array<tvm::PrimExpr> required = args[3];
  PassInfo pass_info = PassInfo(opt_level, name, required);
  *ret = Sequential(passes, pass_info);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<SequentialNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const SequentialNode*>(ref.get());
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

TVM_REGISTER_GLOBAL("relay._transform.PassContext")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  auto pctx = PassContext::Create();
  int opt_level = args[0];
  int fallback_device = args[1];
  tvm::Array<tvm::PrimExpr> required = args[2];
  tvm::Array<tvm::PrimExpr> disabled = args[3];
  pctx->opt_level = opt_level;
  pctx->fallback_device = fallback_device;
  pctx->required_pass = std::move(required);
  pctx->disabled_pass = std::move(disabled);
  *ret = pctx;
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<PassContextNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const PassContextNode*>(ref.get());
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

TVM_REGISTER_GLOBAL("relay._transform.GetCurrentPassContext")
.set_body_typed(PassContext::Current);

TVM_REGISTER_GLOBAL("relay._transform.EnterPassContext")
.set_body_typed(PassContext::Internal::EnterScope);

TVM_REGISTER_GLOBAL("relay._transform.ExitPassContext")
.set_body_typed(PassContext::Internal::ExitScope);

}  // namespace transform
}  // namespace tvm

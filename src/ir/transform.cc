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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/ir/transform.h>
#include <tvm/node/repr_printer.h>
#include <tvm/node/structural_hash.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/device_api.h>

#include <optional>
#include <queue>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace transform {

using tvm::ReprPrinter;
using tvm::ffi::Any;

TVM_REGISTER_PASS_CONFIG_OPTION("testing.immutable_module", Bool);

struct PassContextThreadLocalEntry {
  /*! \brief The default pass context. */
  PassContext default_context;

  /*! \brief The current pass context. */
  std::stack<PassContext> context_stack;

  PassContextThreadLocalEntry() {
    default_context = PassContext(ffi::make_object<PassContextNode>());
  }
};

/*! \brief Thread local store to hold the pass context. */
static PassContextThreadLocalEntry* PassContextThreadLocalStoreGet() {
  static thread_local PassContextThreadLocalEntry inst;
  return &inst;
}

void PassContext::EnterWithScope() {
  InstrumentEnterPassContext();

  PassContextThreadLocalEntry* entry = PassContextThreadLocalStoreGet();
  entry->context_stack.push(*this);
}

void PassContext::ExitWithScope() {
  PassContextThreadLocalEntry* entry = PassContextThreadLocalStoreGet();
  ICHECK(!entry->context_stack.empty());
  ICHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();

  InstrumentExitPassContext();
}

PassContext PassContext::Current() {
  PassContextThreadLocalEntry* entry = PassContextThreadLocalStoreGet();
  if (!entry->context_stack.empty()) {
    return entry->context_stack.top();
  } else {
    return entry->default_context;
  }
}

// linearly scan the pass array to match pass_name
bool PassArrayContains(const ffi::Array<ffi::String>& pass_array, const std::string& pass_name) {
  for (auto x : pass_array) {
    if (x == pass_name) return true;
  }
  return false;
}

bool PassContext::PassEnabled(const PassInfo& info) const {
  if (PassArrayContains(operator->()->disabled_pass, info->name)) {
    return false;
  }

  if (PassArrayContains(operator->()->required_pass, info->name)) {
    return true;
  }

  return operator->()->opt_level >= info->opt_level;
}

class PassConfigManager {
 public:
  void Register(std::string key, ffi::String value_type_str,
                std::function<ffi::Any(ffi::Any)> legalization) {
    ICHECK_EQ(key2vtype_.count(key), 0U);
    ValueTypeInfo info;
    info.type_str = value_type_str;
    info.legalization = legalization;
    key2vtype_[key] = info;
  }

  // Trying to validate and legalize a config.
  void Legalize(ffi::Map<ffi::String, ffi::Any>* config) {
    std::vector<std::pair<std::string, ffi::Any>> update;
    for (auto [key, value] : *config) {
      auto it = key2vtype_.find(key);
      if (it == key2vtype_.end()) {
        std::ostringstream os;
        os << "AttributeError: Invalid config option \'" << key << "\' candidates are:";
        int counter = 0;
        for (const auto& [key, value] : key2vtype_) {
          os << ' ';
          if (counter++ != 0) os << ',';
          os << key;
        }
        LOG(FATAL) << os.str();
      }
      const auto& info = it->second;

      ICHECK(value != nullptr) << "AttributeError: " << key << " is None";

      ICHECK(info.legalization) << "AttributeError: "
                                << "Config option \'" << key
                                << "\' was defined without a legalization function.";
      auto legalized = info.legalization(value);
      if (!legalized.same_as(value)) {
        update.emplace_back(key, legalized);
      }
    }
    for (auto&& kv : update) {
      config->Set(kv.first, kv.second);
    }
  }

  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::String>> ListConfigs() {
    ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::String>> configs;
    for (const auto& kv : key2vtype_) {
      ffi::Map<ffi::String, ffi::String> metadata;
      metadata.Set("type", kv.second.type_str);
      configs.Set(kv.first, metadata);
    }
    return configs;
  }

  static PassConfigManager* Global() {
    static auto* inst = new PassConfigManager();
    return inst;
  }

 private:
  struct ValueTypeInfo {
    std::string type_str;
    std::function<ffi::Any(ffi::Any)> legalization;
  };

  std::unordered_map<std::string, ValueTypeInfo> key2vtype_;
};

void PassContext::RegisterConfigOption(const char* key, ffi::String value_type_str,
                                       std::function<ffi::Any(ffi::Any)> legalization) {
  PassConfigManager::Global()->Register(key, value_type_str, legalization);
}

ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::String>> PassContext::ListConfigs() {
  return PassConfigManager::Global()->ListConfigs();
}

PassContext PassContext::Create() { return PassContext(ffi::make_object<PassContextNode>()); }

namespace {
struct ClearOnError {
  ffi::Array<instrument::PassInstrument>* instruments{nullptr};

  ~ClearOnError() {
    if (instruments) {
      LOG(INFO) << "Pass instrumentation enter/exti failed.";
      LOG(INFO) << "Disabling pass instrumentation.";
      instruments->clear();
    }
  }
};
struct ExitContextOnError {
  std::vector<instrument::PassInstrument> successes;

  ~ExitContextOnError() {
    for (auto it = successes.rbegin(); it != successes.rend(); it++) {
      LOG(INFO) << (*it)->name << " exiting PassContext ...";
      (*it)->ExitPassContext();
      LOG(INFO) << (*it)->name << " exited PassContext.";
    }
  }
};
}  // namespace

void PassContext::InstrumentEnterPassContext() {
  auto pass_ctx_node = this->operator->();
  if (pass_ctx_node->instruments.defined()) {
    ClearOnError clear_context{&pass_ctx_node->instruments};
    ExitContextOnError exit_context;
    for (instrument::PassInstrument pi : pass_ctx_node->instruments) {
      pi->EnterPassContext();
      exit_context.successes.push_back(pi);
    }
    exit_context.successes.clear();
    clear_context.instruments = nullptr;
  }
}

namespace {

struct ExitPassSuccesses {
  ~ExitPassSuccesses() {
    if (all_initialized) {
      return;
    }

    LOG(INFO) << "Pass instrumentation entering pass context failed.";
    LOG(INFO) << "Disable pass instrumentation.";
    instruments->clear();

    for (auto it = successes.rbegin(); it != successes.rend(); it++) {
      LOG(INFO) << (*it)->name << " exiting PassContext ...";
      (*it)->ExitPassContext();
      LOG(INFO) << (*it)->name << " exited PassContext.";
    }
  }

  bool all_initialized{false};
  std::vector<instrument::PassInstrument> successes;
  ffi::Array<instrument::PassInstrument>* instruments{nullptr};
};
}  // namespace

void PassContext::InstrumentExitPassContext() {
  auto pass_ctx_node = this->operator->();
  if (pass_ctx_node->instruments.defined()) {
    ClearOnError clear_context{&pass_ctx_node->instruments};
    for (instrument::PassInstrument pi : pass_ctx_node->instruments) {
      pi->ExitPassContext();
    }
    clear_context.instruments = nullptr;
  }
}

bool PassContext::InstrumentBeforePass(const IRModule& ir_module, const PassInfo& pass_info) const {
  auto pass_ctx_node = this->operator->();
  if (!pass_ctx_node->instruments.defined()) {
    return true;
  }

  const bool pass_required = PassArrayContains(pass_ctx_node->required_pass, pass_info->name);
  bool should_run = true;
  if (!pass_required) {
    for (instrument::PassInstrument pi : pass_ctx_node->instruments) {
      should_run &= pi->ShouldRun(ir_module, pass_info);
    }
  }

  if (should_run) {
    for (instrument::PassInstrument pi : pass_ctx_node->instruments) {
      pi->RunBeforePass(ir_module, pass_info);
    }
  }
  return should_run;
}

void PassContext::InstrumentAfterPass(const IRModule& ir_module, const PassInfo& pass_info) const {
  auto pass_ctx_node = this->operator->();
  if (pass_ctx_node->instruments.defined()) {
    for (instrument::PassInstrument pi : pass_ctx_node->instruments) {
      pi->RunAfterPass(ir_module, pass_info);
    }
  }
}

IRModule Pass::operator()(IRModule mod) const {
  return this->operator()(std::move(mod), PassContext::Current());
}

IRModule Pass::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const PassNode* node = operator->();
  ICHECK(node != nullptr);
  const PassInfo& pass_info = node->Info();
  if (!pass_ctx.InstrumentBeforePass(mod, pass_info)) {
    DLOG(INFO) << "Skipping pass : " << pass_info->name
               << " with opt level: " << pass_info->opt_level;
    return mod;
  }
  IRModule ret;
  if (pass_ctx->GetConfig<Bool>("testing.immutable_module", Bool(false)).value()) {
    ret = Pass::AssertImmutableModule(mod, node, pass_ctx);
  } else {
    ret = node->operator()(std::move(mod), pass_ctx);
  }
  pass_ctx.InstrumentAfterPass(ret, pass_info);
  return ret;
}

IRModule Pass::AssertImmutableModule(const IRModule& mod, const PassNode* node,
                                     const PassContext& pass_ctx) {
  size_t before_pass_hash = tvm::StructuralHash()(mod);
  IRModule copy_mod = mod;
  IRModule ret = node->operator()(mod, pass_ctx);
  size_t after_pass_hash = tvm::StructuralHash()(copy_mod);
  if (before_pass_hash != after_pass_hash) {
    // The chance of getting a hash conflict between a module and the same module but mutated
    // must be very low.
    LOG_FATAL << "Immutable module has been modified in pass: " << node->Info()->name;
  }
  return ret;
}

/*!
 * \brief Module-level passes are designed to implement global
 * analysis/optimizations, i.e. interprocedural optimizations (IPO), etc. Passes
 * at this level have the full control of a given Relax program including
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
  std::function<IRModule(IRModule, PassContext)> pass_func;

  ModulePassNode() = default;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ModulePassNode>().def_ro("pass_info", &ModulePassNode::pass_info);
  }

  /*!
   * \brief Run a module pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param mod The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("transform.ModulePass", ModulePassNode, PassNode);
};

class ModulePass : public Pass {
 public:
  ModulePass(std::function<IRModule(IRModule, PassContext)> pass_func, PassInfo pass_info);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ModulePass, Pass, ModulePassNode);
};

PassInfo::PassInfo(int opt_level, ffi::String name, tvm::ffi::Array<ffi::String> required,
                   bool traceable) {
  auto pass_info = ffi::make_object<PassInfoNode>();
  pass_info->opt_level = opt_level;
  pass_info->name = std::move(name);
  pass_info->required = std::move(required);
  pass_info->traceable = std::move(traceable);
  data_ = std::move(pass_info);
}

ModulePass::ModulePass(std::function<IRModule(IRModule, PassContext)> pass_func,
                       PassInfo pass_info) {
  auto n = ffi::make_object<ModulePassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

// Module -> Module optimizations.
IRModule ModulePassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  DiagnosticContext previous = DiagnosticContext::Default(mod);

  if (pass_ctx->diag_ctx) {
    DiagnosticContext tmp = pass_ctx->diag_ctx.value();
    pass_ctx->diag_ctx = previous;
    previous = tmp;
  } else {
    pass_ctx->diag_ctx = previous;
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  const PassInfo& pass_info = Info();
  ICHECK(mod.defined()) << "The input module must be set.";

  VLOG_CONTEXT << pass_info->name;
  VLOG(0) << "Executing module pass with opt level: " << pass_info->opt_level;

  mod = pass_func(std::move(mod), pass_ctx);

  ICHECK(mod.defined()) << "The return value of a module pass must be set.";

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  pass_ctx->diag_ctx.value().Render();
  pass_ctx->diag_ctx = previous;

  return mod;
}

Sequential::Sequential(tvm::ffi::Array<Pass> passes, PassInfo pass_info) {
  auto n = ffi::make_object<SequentialNode>();
  n->passes = std::move(passes);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

Sequential::Sequential(tvm::ffi::Array<Pass> passes, ffi::String name) {
  auto n = ffi::make_object<SequentialNode>();
  n->passes = std::move(passes);
  PassInfo pass_info = PassInfo(0, std::move(name), {}, /* traceable */ false);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

const SequentialNode* Sequential::operator->() const {
  return static_cast<const SequentialNode*>(get());
}

Pass GetPass(const ffi::String& pass_name) {
  std::optional<tvm::ffi::Function> f;
  if (pass_name.operator std::string().find("transform.") != std::string::npos) {
    f = tvm::ffi::Function::GetGlobal(pass_name);
  } else {
    f = tvm::ffi::Function::GetGlobal("transform." + pass_name);
  }
  ICHECK(f.has_value()) << "Cannot use " << pass_name << " to create the pass";
  return (*f)().cast<Pass>();
}

// Safe version of GetPass that returns empty optional instead of throwing
std::optional<Pass> TryGetPass(const ffi::String& pass_name) {
  std::optional<tvm::ffi::Function> f;
  if (pass_name.operator std::string().find("transform.") != std::string::npos) {
    f = tvm::ffi::Function::GetGlobal(pass_name);
  } else {
    f = tvm::ffi::Function::GetGlobal("transform." + pass_name);
  }
  if (!f.has_value()) {
    return std::nullopt;
  }
  return (*f)().cast<Pass>();
}

void SequentialNode::ResolveDependency(const IRModule& mod) {
  // Get the current pass context to check which passes are enabled
  // Note: mod parameter is reserved for future use when dependency resolution
  // might need to consider module-specific information
  (void)mod;  // Suppress unused parameter warning
  PassContext pass_ctx = PassContext::Current();

  // Step 1: Collect all enabled passes from the current list
  std::unordered_map<std::string, Pass> name_to_pass;
  std::vector<Pass> enabled_passes;

  for (const Pass& pass : passes) {
    if (!pass.defined()) {
      continue;
    }
    const PassInfo& pass_info = pass->Info();
    if (pass_ctx.PassEnabled(pass_info)) {
      std::string pass_name = pass_info->name;
      // Avoid duplicates
      if (name_to_pass.find(pass_name) == name_to_pass.end()) {
        name_to_pass[pass_name] = pass;
        enabled_passes.push_back(pass);
      }
    }
  }

  // Step 2: Collect all required passes that are not in the current list
  // We need to do this in multiple passes to handle transitive dependencies
  std::unordered_set<std::string> processed_required;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < enabled_passes.size(); ++i) {
      const PassInfo& pass_info = enabled_passes[i]->Info();
      for (const auto& required_name : pass_info->required) {
        std::string req_name = required_name;
        std::string key = pass_info->name + "->" + req_name;
        if (processed_required.find(key) != processed_required.end()) {
          continue;
        }
        processed_required.insert(key);

        // Check if the required pass is already in our list
        if (name_to_pass.find(req_name) == name_to_pass.end()) {
          // Try to get it from the global registry
          // Use TryGetPass to avoid exceptions when the pass is not registered
          std::optional<Pass> required_pass_opt = TryGetPass(ffi::String(req_name));
          if (required_pass_opt.has_value()) {
            Pass required_pass = required_pass_opt.value();
            const PassInfo& req_pass_info = required_pass->Info();
            if (pass_ctx.PassEnabled(req_pass_info)) {
              name_to_pass[req_name] = required_pass;
              enabled_passes.push_back(required_pass);
              changed = true;
            }
          } else {
            // If we can't get the pass from the registry, we'll skip this dependency
            // This can happen if the required pass is not registered globally
            // It will be resolved at runtime in operator() if needed
            VLOG(0) << "Warning: Cannot resolve required pass '" << req_name << "' for pass '"
                    << pass_info->name
                    << "' from global registry. It will be resolved at runtime if needed.";
          }
        }
      }
    }
  }

  // Step 3: Build dependency graph
  // Map from pass name to its index in enabled_passes
  std::unordered_map<std::string, size_t> name_to_index;
  for (size_t i = 0; i < enabled_passes.size(); ++i) {
    const PassInfo& pass_info = enabled_passes[i]->Info();
    name_to_index[pass_info->name] = i;
  }

  // Build reverse adjacency list: dependents[i] contains indices of passes that depend on pass i
  // This is used for topological sort
  std::vector<std::vector<size_t>> dependents(enabled_passes.size());
  std::vector<size_t> in_degree(enabled_passes.size(), 0);

  for (size_t i = 0; i < enabled_passes.size(); ++i) {
    const PassInfo& pass_info = enabled_passes[i]->Info();
    for (const auto& required_name : pass_info->required) {
      std::string req_name = required_name;
      auto it = name_to_index.find(req_name);
      if (it != name_to_index.end()) {
        // The required pass is in our enabled passes list
        // pass i depends on pass req_idx, so req_idx should come before i
        size_t req_idx = it->second;
        dependents[req_idx].push_back(i);
        in_degree[i]++;
      }
      // If the required pass is not in our list, it will be handled at runtime
    }
  }

  // Step 4: Topological sort using Kahn's algorithm
  std::queue<size_t> queue;
  for (size_t i = 0; i < enabled_passes.size(); ++i) {
    if (in_degree[i] == 0) {
      queue.push(i);
    }
  }

  std::vector<Pass> sorted_passes;
  // Track which passes have been sorted to handle circular dependencies
  std::vector<bool> sorted(enabled_passes.size(), false);

  while (!queue.empty()) {
    size_t current = queue.front();
    queue.pop();

    // In Kahn's algorithm, a node is added to queue only when in_degree becomes 0,
    // which happens exactly once for each node in a DAG, so no need to check visited
    sorted_passes.push_back(enabled_passes[current]);
    sorted[current] = true;

    // Process dependents: passes that depend on the current pass
    for (size_t dependent : dependents[current]) {
      in_degree[dependent]--;
      if (in_degree[dependent] == 0) {
        queue.push(dependent);
      }
    }
  }

  // Check for circular dependencies
  if (sorted_passes.size() != enabled_passes.size()) {
    std::ostringstream os;
    os << "Circular dependency detected in pass sequence. "
       << "Only " << sorted_passes.size() << " out of " << enabled_passes.size()
       << " passes were sorted. Remaining passes will be appended in original order.";
    LOG(WARNING) << os.str();
    // Add remaining passes that weren't sorted (they have circular dependencies)
    for (size_t i = 0; i < enabled_passes.size(); ++i) {
      if (!sorted[i]) {
        sorted_passes.push_back(enabled_passes[i]);
      }
    }
  }

  // Step 5: Update the passes list
  passes = ffi::Array<Pass>(sorted_passes);
}

IRModule SequentialNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  // Resolve dependencies and sort passes using topological sort
  // Note: We need to call ResolveDependency which modifies the passes member,
  // but since SequentialNode is an Object (immutable reference), we can safely
  // modify it here as the actual object data is mutable.
  const_cast<SequentialNode*>(this)->ResolveDependency(mod);

  // Execute passes in the resolved order
  for (const Pass& pass : passes) {
    VLOG(0) << "Running pass " << pass->Info()->name;
    ICHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!pass_ctx.PassEnabled(pass_info)) {
      VLOG(0) << "skipping disabled pass '" << pass_info->name << "'";
      continue;
    }

    // Dependencies are already resolved and sorted by ResolveDependency,
    // so we just execute the pass directly
    mod = pass(std::move(mod), pass_ctx);
  }
  return mod;
}

Pass CreateModulePass(std::function<IRModule(IRModule, PassContext)> pass_func, int opt_level,
                      ffi::String name, tvm::ffi::Array<ffi::String> required, bool traceable) {
  PassInfo pass_info = PassInfo(opt_level, name, required, traceable);
  return ModulePass(std::move(pass_func), pass_info);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("transform.PassInfo",
           [](int opt_level, ffi::String name, tvm::ffi::Array<ffi::String> required,
              bool traceable) { return PassInfo(opt_level, name, required, traceable); })
      .def_packed("transform.Info", [](ffi::PackedArgs args, ffi::Any* ret) {
        Pass pass = args[0].cast<Pass>();
        *ret = pass->Info();
      });
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PassInfoNode>([](const ObjectRef& ref, tvm::ReprPrinter* p) {
      auto* node = static_cast<const PassInfoNode*>(ref.get());
      p->stream << "The meta data of the pass - ";
      p->stream << "pass name: " << node->name;
      p->stream << ", opt_level: " << node->opt_level;
      if (node->required.empty()) {
        p->stream << ", required passes: []\n";
      } else {
        p->stream << ", required passes: ["
                  << "\n";
        for (const auto& it : node->required) {
          p->stream << it << ", ";
        }
        p->stream << "]\n";
      }
    });

TVM_FFI_STATIC_INIT_BLOCK() {
  PassContextNode::RegisterReflection();
  PassInfoNode::RegisterReflection();
  SequentialNode::RegisterReflection();
  ModulePassNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("transform.MakeModulePass",
           [](ffi::TypedFunction<IRModule(ffi::RValueRef<IRModule>, PassContext)> pass_func,
              PassInfo pass_info) {
             auto wrapped_pass_func = [pass_func](IRModule mod, PassContext ctx) {
               return pass_func(ffi::RValueRef<IRModule>(std::move(mod)), ctx);
             };
             return ModulePass(wrapped_pass_func, pass_info);
           })
      .def("transform.RunPass",
           [](Pass pass, ffi::RValueRef<IRModule> mod) { return pass(*std::move(mod)); });
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ModulePassNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ModulePassNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run Module pass: " << info->name << " at the optimization level "
                << info->opt_level;
    });

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("transform.Sequential", [](ffi::PackedArgs args, ffi::Any* ret) {
    auto passes = args[0].cast<tvm::ffi::Array<Pass>>();
    int opt_level = args[1].cast<int>();
    std::string name = args[2].cast<std::string>();
    auto required = args[3].cast<tvm::ffi::Array<ffi::String>>();
    bool traceable = args[4].cast<bool>();
    PassInfo pass_info = PassInfo(opt_level, name, required, /* traceable */ traceable);
    *ret = Sequential(passes, pass_info);
  });
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SequentialNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const SequentialNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run Sequential pass: " << info->name << " at the optimization level "
                << info->opt_level << ". ";
      p->stream << "The passes will be executed are: [";
      for (const auto& it : node->passes) {
        const PassInfo pass_info = it->Info();
        p->stream << pass_info->name << " ";
      }
      p->stream << "]";
    });

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "transform.PassContext",
      [](int opt_level, ffi::Array<ffi::String> required, ffi::Array<ffi::String> disabled,
         ffi::Array<instrument::PassInstrument> instruments,
         ffi::Optional<ffi::Map<ffi::String, ffi::Any>> config) {
        auto pctx = PassContext::Create();
        pctx->opt_level = opt_level;

        pctx->required_pass = std::move(required);
        pctx->disabled_pass = std::move(disabled);
        pctx->instruments = std::move(instruments);

        if (config.defined()) {
          pctx->config = config.value();
        }
        PassConfigManager::Global()->Legalize(&(pctx->config));
        return pctx;
      });
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PassContextNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PassContextNode*>(ref.get());
      p->stream << "Pass context information: "
                << "\n";
      p->stream << "\topt_level: " << node->opt_level << "\n";

      p->stream << "\trequired passes: " << node->required_pass << "\n";
      p->stream << "\tdisabled passes: " << node->disabled_pass << "\n";
      p->stream << "\tinstruments: " << node->instruments << "\n";

      p->stream << "\tconfig: " << node->config << "\n";
    });

class PassContext::Internal {
 public:
  static void EnterScope(PassContext pass_ctx) { pass_ctx.EnterWithScope(); }

  static void ExitScope(PassContext pass_ctx) { pass_ctx.ExitWithScope(); }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("transform.GetCurrentPassContext", PassContext::Current)
      .def("transform.EnterPassContext", PassContext::Internal::EnterScope)
      .def("transform.ExitPassContext", PassContext::Internal::ExitScope)
      .def("transform.OverrideInstruments",
           [](PassContext pass_ctx, ffi::Array<instrument::PassInstrument> instruments) {
             pass_ctx.InstrumentExitPassContext();
             pass_ctx->instruments = instruments;
             pass_ctx.InstrumentEnterPassContext();
           });
}

Pass PrintIR(ffi::String header) {
  auto pass_func = [header](IRModule mod, const PassContext& ctx) {
    LOG(INFO) << "PrintIR(" << header << "):\n" << mod;
    return mod;
  };
  return CreateModulePass(pass_func, 0, "PrintIR", {}, /* traceable */ false);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("transform.PrintIR", PrintIR)
      .def("transform.ListConfigs", PassContext::ListConfigs);
}

}  // namespace transform
}  // namespace tvm

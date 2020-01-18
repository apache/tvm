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
 *  Compile executable modules.
 * \file build_module.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/build_module.h>
#include <tvm/top/operation.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/codegen.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <mutex>
#include <stack>

namespace tvm {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;
using tir::LoweredFunc;

TVM_REGISTER_NODE_TYPE(GenericFuncNode);

bool LLVMEnabled() {
  const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.build_llvm");
  return pf != nullptr;
}

/*! \return The default host target for a given device target */
Target DefaultTargetHost(Target target) {
  if (target.defined() && target->device_type == kDLCPU) {
    return target;
  } else {
    if (LLVMEnabled()) {
      return target::llvm();
    } else {
      return target::stackvm();
    }
  }
}

tir::Buffer BufferWithOffsetAlignment(Array<PrimExpr> shape,
                                      DataType dtype,
                                      std::string name,
                                      int data_alignment,
                                      int offset_factor,
                                      bool compact) {
  auto data = tir::Var(name, DataType::Handle());
  bool has_any = false;
  if (!compact) {
    for (const auto& it : shape) {
      if (it.as<tir::VarNode>()) {
        has_any = true;
        break;
      }
    }
  }
  tir::BufferType buffer_type = has_any ? tir::kAutoBroadcast : tir::kDefault;

  PrimExpr elem_offset;
  if (offset_factor != 0) {
    elem_offset = tir::Var(name + "_elem_offset", shape[0].dtype());
  } else {
    elem_offset = PrimExpr();
  }

  return tir::BufferNode::make(data, dtype, shape, Array<PrimExpr>(), elem_offset, name, "",
    data_alignment, offset_factor, buffer_type);
}

void GetBinds(const Array<top::Tensor>& args,
              bool compact,
              const std::unordered_map<top::Tensor, tir::Buffer>& binds,
              Map<top::Tensor, tir::Buffer>* out_binds,
              Array<ObjectRef>* out_arg_list,
              const BuildConfig& config) {
  *out_binds = binds;

  for (const auto &x : args) {
    if (out_binds->find(x) == out_binds->end()) {
      auto buf = BufferWithOffsetAlignment(x->shape, x->dtype, x->op->name,
        config->data_alignment, config->offset_factor, compact);
      out_binds->Set(x, buf);
      out_arg_list->push_back(buf);
    } else {
      out_arg_list->push_back((*out_binds)[x]);
    }
  }
}

/*!
* \brief Build a Stmt given a schedule, args and binds. This function runs the IR passes.
* \param sch The schedule to build.
* \param args The arguments for the schedule.
* \param binds Buffer assignments.
* \param loop_partition True if the LoopPartition pass should be included.
* \param out_arg_list Returns the arguments for the Stmt.
* \param config The build configuration.
* \return The built Stmt.
*/
tir::Stmt BuildStmt(top::Schedule sch,
                    const Array<top::Tensor>& args,
                    const std::unordered_map<top::Tensor, tir::Buffer>& binds,
                    bool loop_partition,
                    Array<ObjectRef> *out_arg_list,
                    const BuildConfig& config) {
  sch = sch.normalize();

  // Phase 0
  auto bounds = top::InferBound(sch);
  auto stmt = top::ScheduleOps(sch, bounds, false);
  stmt = tir::InjectPrefetch(stmt);

  bool compact = tir::VerifyCompactBuffer(stmt);
  Map<top::Tensor, tir::Buffer> out_binds;
  GetBinds(args, compact, binds, &out_binds, out_arg_list, config);

  // Phase 1
  stmt = tir::StorageFlatten(stmt, out_binds, 64,
                            config->instrument_bound_checkers);
  stmt = tir::CanonicalSimplify(stmt);
  if (loop_partition) {
    stmt = tir::LoopPartition(stmt, config->partition_const_loop);
  }
  if (config->disable_vectorize) {
    stmt = tir::SkipVectorize(stmt);
  } else {
    stmt = tir::VectorizeLoop(stmt);
  }
  stmt = tir::InjectVirtualThread(stmt);
  stmt = tir::InjectDoubleBuffer(stmt, config->double_buffer_split_loop);
  stmt = tir::StorageRewrite(stmt);
  stmt = tir::UnrollLoop(stmt, config->auto_unroll_max_step, config->auto_unroll_max_depth,
    config->auto_unroll_max_extent, config->unroll_explicit);

  // Phase 2
  stmt = tir::Simplify(stmt);
  stmt = tir::RemoveNoOp(stmt);

  if (!(config->disable_select_rewriting))
    stmt = tir::RewriteUnsafeSelect(stmt);

  if (config->instrument_bound_checkers)
    stmt = tir::InstrumentBoundCheckers(stmt);

  return stmt;
}

Array<LoweredFunc> lower(top::Schedule sch,
                         const Array<top::Tensor>& args,
                         const std::string& name,
                         const std::unordered_map<top::Tensor, tir::Buffer>& binds,
                         const BuildConfig& config) {
  Array<ObjectRef> out_arg_list;
  auto stmt = BuildStmt(sch, args, binds, true, &out_arg_list, config);
  return Array<LoweredFunc>({ tir::MakeAPI(stmt, name, out_arg_list, 0, config->restricted_func) });
}

Array<Array<LoweredFunc> > split_dev_host_funcs(const Array<LoweredFunc>& funcs,
                                                const Target& target,
                                                const Target& target_host,
                                                const BuildConfig& config) {
  std::unordered_set<std::string> all_names;
  for (const auto& x : funcs) {
    CHECK(all_names.count(x->name) == 0)
        << "Duplicate function name " << x->name;
    all_names.insert(x->name);
  }

  Array<LoweredFunc> fhost;
  Array<LoweredFunc> fdevice;

  for (const auto& x : funcs) {
    CHECK(tir::VerifyMemory(x, target->device_type))
        << "Direct host side access to device memory is detected in "
        << x->func_name() << ". Did you forget to bind?";

    if (x->func_type == tir::kMixedFunc) {
      auto func = x;
      if (config->detect_global_barrier) {
        func = tir::ThreadSync(func, "global");
      }

      func = tir::ThreadSync(func, "shared");
      func = tir::ThreadSync(func, "warp");
      func = tir::LowerThreadAllreduce(func, target->thread_warp_size);
      auto fsplits = tir::SplitHostDevice(func);
      fhost.push_back(fsplits[0]);
      for (auto f = fsplits.begin() + 1; f != fsplits.end(); ++f) {
        fdevice.push_back(*f);
      }
    } else if (x->func_type == tir::kHostFunc) {
      fhost.push_back(x);
    } else if (x->func_type == tir::kDeviceFunc) {
      fdevice.push_back(x);
    } else {
      LOG(FATAL) << "unknown function type " << x->func_type;
    }
  }

  for (size_t i = 0; i < fdevice.size(); i++) {
    auto warp_size = target->thread_warp_size;
    auto func = fdevice[i];
    func = tir::LowerWarpMemory(fdevice[i], warp_size);
    fdevice.Set(i, func);
  }

  auto keys = target->keys();
  bool target_is_gpu = std::find(keys.begin(), keys.end(), "gpu") != keys.end();
  if (target_is_gpu && fdevice.size() == 0) {
    LOG(WARNING) << "Specified target "
                 << target->str()
                 << " but cannot find device code. Did you forget to bind?";
  }

  for (size_t i = 0; i < fdevice.size(); ++i) {
    auto func = fdevice[i];
    func = tir::LowerIntrin(func, target->target_name);
    fdevice.Set(i, func);
  }

  if (target->device_type == target::llvm()->device_type &&
        target_host == target) {
    CHECK(fdevice.empty()) << "No device code should be generated when target "
                           << "and host_target are both llvm target."
                           << "\n";
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    auto func = fhost[i];
    func = tir::BindDeviceType(func, target->device_type);
    func = tir::LowerDeviceStorageAccessInfo(func);
    func = tir::LowerTVMBuiltin(func);
    fhost.Set(i, func);
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    auto func = fhost[i];
    func = tir::LowerIntrin(func, target_host->target_name);
    func = tir::LowerDeviceStorageAccessInfo(func);
    func = tir::CombineContextCall(func);
    fhost.Set(i, func);
  }
  return {fhost, fdevice};
}

// Create a module for a specific device (target). The lowered functions
// associated with the host is returned as well.
runtime::Module DeviceBuild(const Array<LoweredFunc>& fdevice,
                            const Target& target) {
  if (!fdevice.empty()) {
    return codegen::Build(fdevice, target->str());
  } else {
    return runtime::Module(nullptr);
  }
}

// Build for heterogeneous execution.
runtime::Module build(const Map<Target, Array<LoweredFunc>>& inputs,
                      const Target& target_host,
                      const BuildConfig& config) {
  Array<LoweredFunc> fhost_all;
  std::vector<runtime::Module> device_modules;

  Target target_host_val = target_host;
  if (!target_host.defined()) {
    for (const auto& it : inputs) {
      if (it.first->device_type == kDLCPU) {
        target_host_val = it.first;
        break;
      }
    }
  }

  if (!target_host_val.defined()) {
    target_host_val = DefaultTargetHost(target_host_val);
  }

  for (const auto& it : inputs) {
    auto host_dev_funcs =
        split_dev_host_funcs(it.second, it.first, target_host_val, config);
    auto& fhost = host_dev_funcs[0];
    auto& fdevice = host_dev_funcs[1];
    // Get the module for a certain target.
    runtime::Module mdev = DeviceBuild(fdevice, it.first);
    for (const auto& it : fhost) {
      fhost_all.push_back(it);
    }
    device_modules.push_back(mdev);
  }

  runtime::Module mhost = codegen::Build(fhost_all, target_host_val->str());
  // Import all modules
  for (const auto& it : device_modules) {
    if (it.operator->()) {
      mhost.Import(it);
    }
  }
  return mhost;
}

// Build for heterogeneous execution when target is a string.
runtime::Module build(const Map<std::string, Array<LoweredFunc>>& inputs,
                      const Target& target_host,
                      const BuildConfig& config) {
  Map<Target, Array<LoweredFunc>> updated_input;
  for (const auto& it : inputs) {
    auto target = Target::Create(it.first);
    if (target->device_name == "vta") {
      target = Target::Create("ext_dev");
    }
    updated_input.Set(target, it.second);
  }
  return build(updated_input, target_host, config);
}

// Build for homogeneous execution.
runtime::Module build(const Array<LoweredFunc>& funcs,
                      const Target& target,
                      const Target& target_host,
                      const BuildConfig& config) {
  Map<Target, Array<LoweredFunc>> inputs = {{target, funcs}};
  return build(inputs, target_host, config);
}

BuildConfig BuildConfig::Create() {
  return BuildConfig(make_object<BuildConfigNode>());
}

/*! \brief Entry to hold the BuildConfig context stack. */
struct TVMBuildConfigThreadLocalEntry {
  /*! \brief The default build config if the stack is empty */
  BuildConfig default_config;

  /*! \brief The current build config context */
  std::stack<BuildConfig> context_stack;

  TVMBuildConfigThreadLocalEntry() :
      default_config(BuildConfig::Create()) {
  }
};

/*! \brief Thread local store to hold the BuildConfig context stack. */
typedef dmlc::ThreadLocalStore<TVMBuildConfigThreadLocalEntry> TVMBuildConfigThreadLocalStore;

void BuildConfig::EnterWithScope() {
  TVMBuildConfigThreadLocalEntry *entry = TVMBuildConfigThreadLocalStore::Get();
  entry->context_stack.push(*this);
}

void BuildConfig::ExitWithScope() {
  TVMBuildConfigThreadLocalEntry *entry = TVMBuildConfigThreadLocalStore::Get();
  CHECK(!entry->context_stack.empty());
  CHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();
}

tvm::BuildConfig BuildConfig::Current() {
  TVMBuildConfigThreadLocalEntry *entry = TVMBuildConfigThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }

  return entry->default_config;
}

TVM_REGISTER_NODE_TYPE(BuildConfigNode);

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<BuildConfigNode>([](const ObjectRef& node, NodePrinter* p) {
  auto* op = static_cast<const BuildConfigNode*>(node.get());
  p->stream << "build_config(";
  p->stream << "data_alignment=" << op->data_alignment << ", ";
  p->stream << "offset_factor=" << op->offset_factor << ", ";
  p->stream << "double_buffer_split_loop=" << op->double_buffer_split_loop << ", ";
  p->stream << "auto_unroll_max_step=" << op->auto_unroll_max_step << ", ";
  p->stream << "auto_unroll_max_depth=" << op->auto_unroll_max_depth << ", ";
  p->stream << "auto_unroll_max_extent=" << op->auto_unroll_max_extent << ", ";
  p->stream << "unroll_explicit=" << op->unroll_explicit << ", ";
  p->stream << "restricted_func=" << op->restricted_func << ", ";
  p->stream << "detect_global_barrier=" << op->detect_global_barrier << ", ";
  p->stream << "partition_const_loop=" << op->partition_const_loop << ", ";
  p->stream << "dump_pass_ir=" << op->dump_pass_ir << ", ";
  p->stream << "instrument_bound_checkers=" << op->instrument_bound_checkers << ", ";
  p->stream << "disable_select_rewriting=" << op->disable_select_rewriting;
  p->stream << "disable_vectorize=" << op->disable_vectorize;
  p->stream << "disable_assert=" << op->disable_assert;
  p->stream << ")";
});

struct GenericFunc::Manager {
  std::unordered_map<std::string, GenericFunc> fmap;
  // mutex
  std::mutex mutex;

  Manager() {
  }

  static Manager* Global() {
    static Manager inst;
    return &inst;
  }
};

GenericFunc GenericFunc::Get(const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex>(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) {
    auto f = make_object<GenericFuncNode>();
    f->name_ = name;
    auto gf = GenericFunc(f);
    m->fmap[name] = gf;
    return gf;
  } else {
    return it->second;
  }
}

void GenericFunc::RegisterGenericFunc(GenericFunc func, const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex>(m->mutex);
  auto it = m->fmap.find(name);
  CHECK(it == m->fmap.end()) << "GenericFunc already registered " << name;
  func->name_ = name;
  m->fmap[name] = func;
}

GenericFunc& GenericFunc::set_default(const PackedFunc value,
                                      bool allow_override) {
  auto node = static_cast<GenericFuncNode*>(operator->());
  if (!allow_override) {
    CHECK(node->generic_func_ == nullptr)
      << "Generic function already registered for " << node->name_;
  }
  node->generic_func_ = value;
  return *this;
}

GenericFunc& GenericFunc::register_func(const std::vector<std::string>& tags,
                                        const PackedFunc value,
                                        bool allow_override) {
  for (auto &t : tags) {
    if (!allow_override) {
      auto iter = (*this)->dispatch_dict_.find(t);
      CHECK(iter == (*this)->dispatch_dict_.end())
        << "Tag " << t << " already registered for schedule factory " << (*this)->name_;
    }
    (*this)->dispatch_dict_[t] = value;
  }
  return *this;
}

void GenericFunc::CallPacked(TVMArgs args, TVMRetValue* ret) const {
  auto node = static_cast<const GenericFuncNode*>(get());
  auto target = Target::Current(true);
  PackedFunc func;

  if (target.defined()) {
    for (auto &k : target->keys()) {
      auto iter = node->dispatch_dict_.find(k);
      if (iter != node->dispatch_dict_.end()) {
        func = iter->second;
        break;
      }
    }
  }

  if (func == nullptr) {
    CHECK(node->generic_func_ != nullptr) << "No generic function registered for " << node->name_;
    func = node->generic_func_;
  }

  func.CallPacked(args, ret);
}

TVM_REGISTER_GLOBAL("_GetCurrentBuildConfig")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = BuildConfig::Current();
  });

class BuildConfig::Internal {
 public:
  static void EnterScope(BuildConfig target) {
    target.EnterWithScope();
  }
  static void ExitScope(BuildConfig target) {
    target.ExitWithScope();
  }
};

TVM_REGISTER_GLOBAL("_EnterBuildConfigScope")
.set_body_typed(BuildConfig::Internal::EnterScope);

TVM_REGISTER_GLOBAL("_ExitBuildConfigScope")
.set_body_typed(BuildConfig::Internal::ExitScope);

TVM_REGISTER_GLOBAL("_BuildConfigSetAddLowerPass")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  BuildConfig cfg = args[0];
  std::vector< std::pair<int, PackedFunc> > add_lower_pass;
  CHECK_EQ(args.size() % 2, 1);
  for (int i = 1; i < args.size(); i += 2) {
    add_lower_pass.push_back(std::make_pair(
      args[i].operator int(),
      args[i + 1].operator tvm::runtime::PackedFunc()));
  }
  cfg->add_lower_pass = add_lower_pass;
  });

TVM_REGISTER_GLOBAL("_BuildConfigGetAddLowerPassInfo")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  // Return one of the following:
  //  * Size of add_lower_pass if num_args == 1
  //  * Phase index of pass if args are (config, index, true)
  //  * Function of pass if args are (config, index, false)
  BuildConfig cfg = args[0];
  if (args.num_args == 1) {
    *ret = static_cast<int64_t>(cfg->add_lower_pass.size());
  } else {
    int index = args[1];
    bool get_phase = args[2];
    auto item = cfg->add_lower_pass[index];
    if (get_phase) {
      *ret = item.first;
    } else {
      *ret = item.second;
    }
  }
  });

TVM_REGISTER_GLOBAL("_GenericFuncCreate")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = GenericFunc(make_object<GenericFuncNode>());
  });

TVM_REGISTER_GLOBAL("_GenericFuncGetGlobal")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string func_name = args[0];
  *ret = GenericFunc::Get(func_name);
  });

TVM_REGISTER_GLOBAL("_GenericFuncSetDefault")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  GenericFunc generic_func = args[0];
  // Intentionally copy and not de-allocate it, to avoid free pyobject during shutdown
  PackedFunc* func = new PackedFunc(args[1].operator PackedFunc());
  bool allow_override = args[2];

  generic_func
    .set_default(*func, allow_override);
  });

TVM_REGISTER_GLOBAL("_GenericFuncRegisterFunc")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  GenericFunc generic_func = args[0];
  // Intentionally copy and not de-allocate it, to avoid free pyobject during shutdown
  PackedFunc* func = new PackedFunc(args[1].operator PackedFunc());
  Array<PrimExpr> tags = args[2];
  bool allow_override = args[3];

  std::vector<std::string> tags_vector;
  for (auto& tag : tags) {
    tags_vector.push_back(tag.as<tvm::tir::StringImmNode>()->value);
  }

  generic_func
    .register_func(tags_vector, *func, allow_override);
  });

TVM_REGISTER_GLOBAL("_GenericFuncCallFunc")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  GenericFunc generic_func = args[0];
  TVMArgs func_args(&args.values[1], &args.type_codes[1], args.num_args - 1);

  generic_func
    .CallPacked(func_args, ret);
  });

TVM_REGISTER_GLOBAL("_GetCurrentTarget")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  bool allow_not_defined = args[0];
  *ret = Target::Current(allow_not_defined);
  });

class Target::Internal {
 public:
  static void EnterScope(Target target) {
    target.EnterWithScope();
  }
  static void ExitScope(Target target) {
    target.ExitWithScope();
  }
};

TVM_REGISTER_GLOBAL("_EnterTargetScope")
.set_body_typed(Target::Internal::EnterScope);

TVM_REGISTER_GLOBAL("_ExitTargetScope")
.set_body_typed(Target::Internal::ExitScope);

}  // namespace tvm

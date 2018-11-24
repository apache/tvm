/*!
 *  Copyright (c) 2017 by Contributors
 *  Compile executable modules.
 * \file build_module.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/build_module.h>
#include <tvm/operation.h>
#include <tvm/ir_pass.h>
#include <tvm/codegen.h>

#include <algorithm>
#include <mutex>
#include <stack>

namespace tvm {

TVM_REGISTER_NODE_TYPE(TargetNode);

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<TargetNode>([](const TargetNode *op, IRPrinter *p) {
  p->stream << op->str();
  });


/*!
* \brief Construct a Target node from the given name and options.
* \param target_name The major target name. Should be one of
* {"llvm", "cuda", "opencl", "metal", "rocm", "stackvm", "opengl", "ext_dev"}
* \param options Additional options appended to the target
* \return The constructed Target
*/
Target CreateTarget(const std::string& target_name,
                    const std::vector<std::string>& options) {
  auto target = Target(make_node<TargetNode>());
  auto t = static_cast<TargetNode*>(target.node_.get());

  t->target_name = target_name;

  std::string libs_flag = "-libs=";
  std::string device_flag = "-device=";
  for (auto& item : options) {
    t->options_array.push_back(ir::StringImm::make(item));

    if (item.find(libs_flag) == 0) {
      std::stringstream ss(item.substr(libs_flag.length()));
      std::string lib_item;
      while (std::getline(ss, lib_item, ',')) {
        t->libs_array.push_back(ir::StringImm::make(lib_item));
      }
    } else if (item.find(device_flag) == 0) {
      t->device_name = item.substr(device_flag.length());
    }
  }

  if (t->device_name.length() > 0) {
    t->keys_array.push_back(ir::StringImm::make(t->device_name));
  }
  t->device_type = kDLCPU;
  t->thread_warp_size = 1;
  if (target_name == "llvm") {
    t->keys_array.push_back(ir::StringImm::make("cpu"));
  } else if (target_name == "cuda" || target_name == "nvptx") {
    t->device_type = kDLGPU;
    t->keys_array.push_back(ir::StringImm::make("cuda"));
    t->keys_array.push_back(ir::StringImm::make("gpu"));
    t->max_num_threads = 512;
    t->thread_warp_size = 32;
  } else if (target_name == "rocm" || target_name == "opencl") {
    // For now assume rocm schedule for opencl
    if (target_name == "opencl") {
      t->device_type = kDLOpenCL;
    } else {
      t->device_type = kDLROCM;
    }
    t->keys_array.push_back(ir::StringImm::make(target_name));
    t->keys_array.push_back(ir::StringImm::make("gpu"));
    t->max_num_threads = 256;
    if (t->device_name == "intel_graphics") {
      t->thread_warp_size = 16;
    }
  } else if (target_name == "metal" || target_name == "vulkan") {
    if (target_name == "metal") {
      t->device_type = kDLMetal;
    } else {
      t->device_type = kDLVulkan;
    }
    t->keys_array.push_back(ir::StringImm::make(target_name));
    t->keys_array.push_back(ir::StringImm::make("gpu"));
    t->max_num_threads = 256;
  } else if (target_name == "sdaccel") {
    t->device_type = kDLOpenCL;
    t->keys_array.push_back(ir::StringImm::make("sdaccel"));
    t->keys_array.push_back(ir::StringImm::make("hls"));
  } else if (target_name == "aocl" || target_name == "aocl_sw_emu") {
    t->device_type = kDLAOCL;
    t->keys_array.push_back(ir::StringImm::make("aocl"));
    t->keys_array.push_back(ir::StringImm::make("hls"));
  } else if (target_name == "opengl") {
    t->device_type = kOpenGL;
    t->keys_array.push_back(ir::StringImm::make("opengl"));
  } else if (target_name == "stackvm") {
    t->device_type = kDLCPU;
  } else if (target_name == "ext_dev") {
    t->device_type = kDLExtDev;
  } else {
    LOG(ERROR) << "Unknown target name " << target_name;
    return target::stackvm();
  }

  return target;
}

TVM_REGISTER_API("_TargetCreate")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string target_name = args[0];
  std::vector<std::string> options;
  for (int i = 1; i < args.num_args; ++i) {
    std::string arg = args[i];
    options.push_back(arg);
  }

  *ret = CreateTarget(target_name, options);
  });

TVM_REGISTER_API("_TargetFromString")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string target_str = args[0];

  *ret = Target::create(target_str);
  });

std::vector<std::string> TargetNode::keys() const {
  std::vector<std::string> result;
  for (auto& expr : keys_array) {
    result.push_back(expr.as<ir::StringImm>()->value);
  }
  return result;
}

std::vector<std::string> TargetNode::options() const {
  std::vector<std::string> result;
  for (auto& expr : options_array) {
    result.push_back(expr.as<ir::StringImm>()->value);
  }
  return result;
}

std::unordered_set<std::string> TargetNode::libs() const {
  std::unordered_set<std::string> result;
  for (auto& expr : libs_array) {
    result.insert(expr.as<ir::StringImm>()->value);
  }
  return result;
}

std::string TargetNode::str() const {
  std::ostringstream result;
  result << target_name;
  for (const auto &x : options()) {
    result << " " << x;
  }
  return result.str();
}


bool StartsWith(const std::string& str, const std::string& pattern) {
  return str.compare(0, pattern.length(), pattern) == 0;
}

std::string GetDeviceName(const std::string& target_str) {
  std::istringstream ss(target_str);
  std::string target_name;
  ss >> target_name;

  std::string item;
  while (ss >> item) {
    if (StartsWith(item, "-device=")) {
      return item.substr(std::string("-device=").length());
    }
  }

  return "";
}

Target Target::create(const std::string& target_str) {
  if (target_str.length() == 0) {
    LOG(ERROR) << "target_str must not be empty";
  }

  std::istringstream ss(target_str);
  std::string target_name;

  ss >> target_name;
  auto device_name = GetDeviceName(target_str);

  std::vector<std::string> options;
  std::string item;
  while (ss >> item) {
    options.push_back(item);
  }

  return CreateTarget(target_name, options);
}

/*! \brief Entry to hold the Target context stack. */
struct TVMTargetThreadLocalEntry {
  /*! \brief The current target context */
  std::stack<tvm::Target> context_stack;

  TVMTargetThreadLocalEntry() {
  }
};

/*! \brief Thread local store to hold the Target context stack. */
typedef dmlc::ThreadLocalStore<TVMTargetThreadLocalEntry> TVMTargetThreadLocalStore;

void Target::EnterTargetScope(const tvm::Target& target) {
  TVMTargetThreadLocalEntry *entry = TVMTargetThreadLocalStore::Get();
  entry->context_stack.push(target);
}

void Target::ExitTargetScope() {
  TVMTargetThreadLocalEntry *entry = TVMTargetThreadLocalStore::Get();
  entry->context_stack.pop();
}

tvm::Target Target::current_target(bool allow_not_defined) {
  TVMTargetThreadLocalEntry *entry = TVMTargetThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }
  CHECK(allow_not_defined)
    << "Target context required. Please set it by constructing a TargetContext";

  return Target();
}

namespace target {
std::vector<std::string> MergeOptions(std::vector<std::string> opts,
                                             const std::vector<std::string>& new_opts) {
  opts.insert(opts.end(), new_opts.begin(), new_opts.end());
  return opts;
}

Target llvm(const std::vector<std::string>& options) {
  return CreateTarget("llvm", options);
}

Target cuda(const std::vector<std::string>& options) {
  return CreateTarget("cuda", options);
}

Target rocm(const std::vector<std::string>& options) {
  return CreateTarget("rocm", options);
}

Target opencl(const std::vector<std::string>& options) {
  return CreateTarget("opencl", options);
}

Target metal(const std::vector<std::string>& options) {
  return CreateTarget("metal", options);
}

Target mali(const std::vector<std::string>& options) {
  return CreateTarget("opencl", MergeOptions(options, {
    "-device=mali"
  }));
}

Target intel_graphics(const std::vector<std::string>& options) {
  return CreateTarget("opencl", MergeOptions(options, {
    "-device=intel_graphics"
  }));
}

Target stackvm(const std::vector<std::string>& options) {
  return CreateTarget("stackvm", options);
}
}  // namespace target

bool LLVMEnabled() {
  const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.build_llvm");
  return pf != nullptr;
}

/*! \return The default host target for a given device target */
Target DefaultTargetHost(Target target) {
  if (target->device_type == kDLCPU) {
    return target;
  } else {
    if (LLVMEnabled()) {
      return target::llvm();
    } else {
      return target::stackvm();
    }
  }
}

Buffer BufferWithOffsetAlignment(Array<Expr> shape,
                                 Type dtype,
                                 std::string name,
                                 int data_alignment,
                                 int offset_factor) {
  auto data = Var(name, Handle());

  Expr elem_offset;
  if (offset_factor != 0) {
    elem_offset = Var(name + "_elem_offset", shape[0].type());
  } else {
    elem_offset = Expr();
  }

  return BufferNode::make(data, dtype, shape, Array<Expr>(), elem_offset, name, "",
    data_alignment, offset_factor);
}

void GetBinds(const Array<Tensor>& args,
              const std::unordered_map<Tensor, Buffer>& binds,
              Map<Tensor, Buffer>* out_binds,
              Array<NodeRef>* out_arg_list,
              const BuildConfig& config) {
  *out_binds = binds;

  for (const auto &x : args) {
    if (out_binds->find(x) == out_binds->end()) {
      auto buf = BufferWithOffsetAlignment(x->shape, x->dtype, x->op->name,
        config->data_alignment, config->offset_factor);
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
Stmt BuildStmt(Schedule sch,
               const Array<Tensor>& args,
               const std::unordered_map<Tensor, Buffer>& binds,
               bool loop_partition,
               Array<NodeRef> *out_arg_list,
               const BuildConfig& config) {
  Map<Tensor, Buffer> out_binds;
  GetBinds(args, binds, &out_binds, out_arg_list, config);

  sch = sch.normalize();

  // Phase 0
  auto bounds = schedule::InferBound(sch);
  auto stmt = schedule::ScheduleOps(sch, bounds, false);
  stmt = ir::InjectPrefetch(stmt);

  // Phase 1
  stmt = ir::StorageFlatten(stmt, out_binds, 64);
  stmt = ir::CanonicalSimplify(stmt);
  if (loop_partition) {
    stmt = ir::LoopPartition(stmt, config->partition_const_loop);
  }
  stmt = ir::VectorizeLoop(stmt);
  stmt = ir::InjectVirtualThread(stmt);
  stmt = ir::InjectDoubleBuffer(stmt, config->double_buffer_split_loop);
  stmt = ir::StorageRewrite(stmt);
  stmt = ir::UnrollLoop(stmt, config->auto_unroll_max_step, config->auto_unroll_max_depth,
    config->auto_unroll_max_extent, config->unroll_explicit);

  // Phase 2
  stmt = ir::Simplify(stmt);
  stmt = ir::LowerStorageAccessInfo(stmt);
  stmt = ir::RemoveNoOp(stmt);
  stmt = ir::RewriteUnsafeSelect(stmt);

  return stmt;
}

Array<LoweredFunc> lower(Schedule sch,
                         const Array<Tensor>& args,
                         const std::string& name,
                         const std::unordered_map<Tensor, Buffer>& binds,
                         const BuildConfig& config) {
  Array<NodeRef> out_arg_list;
  auto stmt = BuildStmt(sch, args, binds, true, &out_arg_list, config);
  return Array<LoweredFunc>({ ir::MakeAPI(stmt, name, out_arg_list, 0, config->restricted_func) });
}

runtime::Module build(const Array<LoweredFunc>& funcs,
                      const Target& target,
                      const Target& target_host,
                      const BuildConfig& config) {
  std::unordered_set<std::string> all_names;
  for (const auto &x : funcs) {
    CHECK(all_names.count(x->name) == 0) << "Duplicate function name " << x->name;
    all_names.insert(x->name);
  }

  auto target_host_val = target_host.defined() ? target_host : DefaultTargetHost(target);

  Array<LoweredFunc> fhost;
  Array<LoweredFunc> fdevice;

  for (const auto& x : funcs) {
    CHECK(ir::VerifyMemory(x, target->device_type))
        << "Direct host side access to device memory is detected in " << x->func_name()
        << ". Did you forget to bind?";

    if (x->func_type == kMixedFunc) {
      auto func = x;
      if (config->detect_global_barrier) {
        func = ir::ThreadSync(func, "global");
      }

      func = ir::ThreadSync(func, "shared");
      func = ir::LowerThreadAllreduce(func, target->thread_warp_size);
      auto fsplits = ir::SplitHostDevice(func);
      fhost.push_back(fsplits[0]);
      for (auto f = fsplits.begin() + 1; f != fsplits.end(); ++f) {
        fdevice.push_back(*f);
      }
    } else if (x->func_type == kHostFunc) {
      fhost.push_back(x);
    } else if (x->func_type == kDeviceFunc) {
      fdevice.push_back(x);
    } else {
      LOG(FATAL) << "unknown function type " << x->func_type;
    }
  }

  auto keys = target->keys();
  bool target_is_gpu =
    std::find(keys.begin(), keys.end(), "gpu") != keys.end();
  if (target_is_gpu && fdevice.size() == 0) {
    LOG(WARNING) << "Specified target " + target->str() +
      " but cannot find device code. Did you forget to bind?";
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    auto func = fhost[i];
    func = ir::BindDeviceType(func, target->device_type);
    func = ir::LowerTVMBuiltin(func);
    fhost.Set(i, func);
  }


  for (size_t i = 0; i < fdevice.size(); ++i) {
    auto func = fdevice[i];
    func = ir::LowerIntrin(func, target->target_name);
    fdevice.Set(i, func);
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    auto func = fhost[i];
    func = ir::LowerIntrin(func, target_host_val->target_name);
    func = ir::CombineContextCall(func);
    fhost.Set(i, func);
  }

  auto mhost = codegen::Build(fhost, target_host_val->str());

  if (fdevice.size() > 0) {
    auto mdev = codegen::Build(fdevice, target->str());
    mhost.Import(mdev);
  }

  return mhost;
}

BuildConfig build_config() {
  return BuildConfig(make_node<BuildConfigNode>());
}

/*! \brief Entry to hold the BuildConfig context stack. */
struct TVMBuildConfigThreadLocalEntry {
  /*! \brief The default build config if the stack is empty */
  tvm::BuildConfig default_config;

  /*! \brief The current build config context */
  std::stack<tvm::BuildConfig> context_stack;

  TVMBuildConfigThreadLocalEntry() :
    default_config(build_config()) {
  }
};

/*! \brief Thread local store to hold the BuildConfig context stack. */
typedef dmlc::ThreadLocalStore<TVMBuildConfigThreadLocalEntry> TVMBuildConfigThreadLocalStore;

void BuildConfig::EnterBuildConfigScope(const tvm::BuildConfig& build_config) {
  TVMBuildConfigThreadLocalEntry *entry = TVMBuildConfigThreadLocalStore::Get();
  entry->context_stack.push(build_config);
}

void BuildConfig::ExitBuildConfigScope() {
  TVMBuildConfigThreadLocalEntry *entry = TVMBuildConfigThreadLocalStore::Get();
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BuildConfigNode>([](const BuildConfigNode *op, IRPrinter *p) {
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
  p->stream << "dump_pass_ir=" << op->dump_pass_ir;
  p->stream << ")";
});

struct GenericFunc::Manager {
  std::unordered_map<std::string, NodePtr<Node> > fmap;
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
    auto f = make_node<GenericFuncNode>();
    f->name_ = name;
    m->fmap[name] = f;
    return GenericFunc(f);
  } else {
    return GenericFunc(it->second);
  }
}

void GenericFunc::RegisterGenericFunc(GenericFunc func, const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex>(m->mutex);
  auto it = m->fmap.find(name);
  CHECK(it == m->fmap.end()) << "GenericFunc already registered " << name;
  func->name_ = name;
  m->fmap[name] = func.node_;
}

GenericFunc& GenericFunc::set_default(const PackedFunc value,
                                           bool allow_override) {
  auto node = static_cast<GenericFuncNode*>(node_.get());
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
  auto node = static_cast<GenericFuncNode*>(node_.get());
  auto target = Target::current_target(true);
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

TVM_REGISTER_API("_GetCurrentBuildConfig")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = BuildConfig::Current();
  });

TVM_REGISTER_API("_EnterBuildConfigScope")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  BuildConfig target = args[0];
  BuildConfig::EnterBuildConfigScope(target);
  });

TVM_REGISTER_API("_ExitBuildConfigScope")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  BuildConfig::ExitBuildConfigScope();
  });

TVM_REGISTER_API("_BuildConfigSetAddLowerPass")
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

TVM_REGISTER_API("_BuildConfigGetAddLowerPassInfo")
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

TVM_REGISTER_API("_GenericFuncCreate")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = GenericFunc(make_node<GenericFuncNode>());
  });

TVM_REGISTER_API("_GenericFuncGetGlobal")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string func_name = args[0];
  *ret = GenericFunc::Get(func_name);
  });

TVM_REGISTER_API("_GenericFuncSetDefault")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  GenericFunc generic_func = args[0];
  // Intentionally copy and not de-allocate it, to avoid free pyobject during shutdown
  PackedFunc* func = new PackedFunc(args[1].operator PackedFunc());
  bool allow_override = args[2];

  generic_func
    .set_default(*func, allow_override);
  });

TVM_REGISTER_API("_GenericFuncRegisterFunc")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  GenericFunc generic_func = args[0];
  // Intentionally copy and not de-allocate it, to avoid free pyobject during shutdown
  PackedFunc* func = new PackedFunc(args[1].operator PackedFunc());
  Array<Expr> tags = args[2];
  bool allow_override = args[3];

  std::vector<std::string> tags_vector;
  for (auto& tag : tags) {
    tags_vector.push_back(tag.as<tvm::ir::StringImm>()->value);
  }

  generic_func
    .register_func(tags_vector, *func, allow_override);
  });

TVM_REGISTER_API("_GenericFuncCallFunc")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  GenericFunc generic_func = args[0];
  TVMArgs func_args(&args.values[1], &args.type_codes[1], args.num_args - 1);

  generic_func
    .CallPacked(func_args, ret);
  });

TVM_REGISTER_API("_GetCurrentTarget")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  bool allow_not_defined = args[0];
  *ret = Target::current_target(allow_not_defined);
  });

TVM_REGISTER_API("_EnterTargetScope")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Target target = args[0];
  Target::EnterTargetScope(target);
  });

TVM_REGISTER_API("_ExitTargetScope")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Target::ExitTargetScope();
  });

}  // namespace tvm

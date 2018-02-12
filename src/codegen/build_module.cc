/*!
 *  Copyright (c) 2017 by Contributors
 *  Compile executable modules.
 * \file build_module.cc
 */
#include <tvm/build_module.h>
#include <tvm/operation.h>
#include <tvm/ir_pass.h>
#include <tvm/codegen.h>

#include <algorithm>
#include <mutex>

namespace tvm {

std::string Target::str() const {
  std::ostringstream result;
  result << target_name;
  for (const auto &x : options) {
    result << " " << x;
  }
  return result.str();
}

Target TargetFromName(const std::string& name) {
  if (name == "llvm") {
    return target::llvm();
  } else if (name == "cuda" || name == "nvptx") {
    return target::cuda();
  } else if (name == "rocm" || name == "opencl") {
    /* For now, assume rocm schedule for opencl */
    auto rocm = target::rocm();
    rocm.target_name = name;  // preserve the original target name for opencl
    return rocm;
  } else if (name == "metal") {
    return target::metal();
  } else if (name == "stackvm" || name == "ext_dev") {
    return target::stackvm();
  } else {
    LOG(ERROR) << "Unknown target name " << name;
    return target::stackvm();
  }
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

  auto result = device_name == "rasp" ?
    target::rasp() :
    (device_name == "mali" ? target::mali() :
    TargetFromName(target_name));

  std::string item;
  while (ss >> item) {
    result.options.push_back(item);
  }

  return result;
}

namespace target {
Target llvm() {
  std::vector<std::string> keys({ "llvm", "cpu" });
  std::vector<std::string> options;
  return Target("llvm", kDLCPU, 512, 1, keys, options,
           std::unordered_set<std::string>());
}

Target cuda() {
  std::vector<std::string> keys({ "cuda", "gpu" });
  std::vector<std::string> options;
  return Target("cuda", kDLGPU, 512, 32, keys, options,
           std::unordered_set<std::string>());
}

Target rocm() {
  std::vector<std::string> keys({ "rocm", "gpu" });
  std::vector<std::string> options;
  return Target("rocm", kDLROCM, 256, 1, keys, options,
           std::unordered_set<std::string>());
}

Target metal() {
  std::vector<std::string> keys({ "gpu" });
  std::vector<std::string> options;
  return Target("metal", kDLMetal, 256, 1, keys, options,
           std::unordered_set<std::string>());
}

Target rasp() {
  std::vector<std::string> keys({ "llvm", "cpu" });
  std::vector<std::string> options({
    "-device=rasp",
    "-mtriple=armv7l-none-linux-gnueabihf",
    "-mcpu=cortex-a53",
    "-mattr=+neon"
  });
  return Target("llvm", kDLCPU, 512, 1, keys, options,
           std::unordered_set<std::string>());
}

Target mali() {
  std::vector<std::string> keys({ "rocm", "gpu" });
  std::vector<std::string> options({
    "-device=mali"
  });
  return Target("opencl", kDLOpenCL, 256, 1, keys, options);
}


Target stackvm() {
  std::vector<std::string> keys({ "stackvm", "cpu" });
  std::vector<std::string> options;
  return Target("stackvm", kDLCPU, 512, 1, keys, options,
           std::unordered_set<std::string>());
}
}  // namespace target

bool LLVMEnabled() {
  const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.build_llvm");
  return pf != nullptr;
}

/*! \return The default host target for a given device target */
Target DefaultTargetHost(Target target) {
  if (target.device_type == kDLCPU) {
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
  auto stmt = schedule::ScheduleOps(sch, bounds, true);
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
                      Target* target_host,
                      const BuildConfig& config) {
  std::unordered_set<std::string> all_names;
  for (const auto &x : funcs) {
    CHECK(all_names.count(x->name) == 0) << "Duplicate function name " << x->name;
    all_names.insert(x->name);
  }

  Target target_host_val = target_host == nullptr ?
    DefaultTargetHost(target) :
    *target_host;

  Array<LoweredFunc> fhost;
  Array<LoweredFunc> fdevice;

  for (const auto &x : funcs) {
    if (x->func_type == kMixedFunc) {
      auto func = x;
      if (config->detect_global_barrier) {
        func = ir::ThreadSync(func, "global");
      }

      func = ir::ThreadSync(func, "shared");
      func = ir::LowerThreadAllreduce(func, target.thread_warp_size);
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

  bool target_is_gpu =
    std::find(target.keys.begin(), target.keys.end(), "gpu") != target.keys.end();
  if (target_is_gpu && fdevice.size() == 0) {
    LOG(WARNING) << "Specified target " + target.str() +
      " but cannot find device code. Did you forget to bind?";
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    auto func = fhost[i];
    func = ir::BindDeviceType(func, target.device_type);
    func = ir::LowerTVMBuiltin(func);
    fhost.Set(i, func);
  }


  for (size_t i = 0; i < fdevice.size(); ++i) {
    auto func = fdevice[i];
    func = ir::LowerIntrin(func, target.target_name);
    fdevice.Set(i, func);
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    auto func = fhost[i];
    func = ir::LowerIntrin(func, target_host_val.target_name);
    func = ir::CombineContextCall(func);
    fhost.Set(i, func);
  }

  auto mhost = codegen::Build(fhost, target_host_val.str());

  if (fdevice.size() > 0) {
    auto mdev = codegen::Build(fdevice, target.str());
    mhost.Import(mdev);
  }

  return mhost;
}

BuildConfig build_config() {
  return BuildConfig(std::make_shared<BuildConfigNode>());
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
  p->stream << "partition_const_loop=" << op->partition_const_loop;
  p->stream << ")";
});

struct GenericFunc::Manager {
  // map storing the functions.
  // We delibrately used raw pointer
  // This is because PackedFunc can contain callbacks into the host languge(python)
  // and the resource can become invalid because of indeterminstic order of destruction.
  // The resources will only be recycled during program exit.
  std::unordered_map<std::string, GenericFunc*> fmap;
  // mutex
  std::mutex mutex;

  Manager() {
  }

  static Manager* Global() {
    static Manager inst;
    return &inst;
  }
};

GenericFunc& GenericFunc::Get(const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex>(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) {
    GenericFunc* f = new GenericFunc();
    f->name_ = name;
    m->fmap[name] = f;
    return *f;
  } else {
    return *it->second;
  }
}

GenericFunc& GenericFunc::set_generic_func(const PackedFunc value,
                                           bool allow_override) {
  if (!allow_override) {
    CHECK(generic_func_ == nullptr) << "Generic function already registered for " << name_;
  }
  this->generic_func_ = value;
  return *this;
}

GenericFunc& GenericFunc::register_func(const std::vector<std::string>& tags,
                                        const PackedFunc value,
                                        bool allow_override) {
  for (auto &t : tags) {
    if (!allow_override) {
      auto iter = dispatch_dict_.find(t);
      CHECK(iter == dispatch_dict_.end())
        << "Tag " << t << " already registered for schedule factory " << name_;
    }
    dispatch_dict_[t] = value;
  }
  return *this;
}

void GenericFunc::invoke_func(const tvm::Target& target, TVMArgs args, TVMRetValue* ret) const {
  PackedFunc func;
  for (auto &k : target.keys) {
    auto iter = dispatch_dict_.find(k);
    if (iter != dispatch_dict_.end()) {
      func = iter->second;
      break;
    }
  }
  if (func == nullptr) {
    CHECK(generic_func_ != nullptr) << "No generic function registered for " << name_;
    func = generic_func_;
  }

  std::vector<TVMValue> values;
  std::vector<int> type_codes;

  auto target_str = target.str();
  TVMValue target_str_value;
  target_str_value.v_str = target_str.c_str();
  values.push_back(target_str_value);
  type_codes.push_back(kStr);

  for (int i = 0; i < args.num_args; ++i) {
    values.push_back(args.values[i]);
    type_codes.push_back(args.type_codes[i]);
  }

  TVMArgs all_args(values.data(), type_codes.data(), values.size());
  func.CallPacked(all_args, ret);
}

TVM_REGISTER_API("_SetGenericFunc")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string func_name = args[0];
  // Intentionally copy and not de-allocate it, to avoid free pyobject during shutdown
  PackedFunc* func = new PackedFunc(args[1].operator PackedFunc());
  bool allow_override = args[2];

  GenericFunc::Get(func_name)
    .set_generic_func(*func, allow_override);
  });

TVM_REGISTER_API("_RegisterFunc")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string func_name = args[0];
  // Intentionally copy and not de-allocate it, to avoid free pyobject during shutdown
  PackedFunc* func = new PackedFunc(args[1].operator PackedFunc());
  Array<Expr> tags = args[2];
  bool allow_override = args[3];

  std::vector<std::string> tags_vector;
  for (auto& tag : tags) {
    tags_vector.push_back(tag.as<tvm::ir::StringImm>()->value);
  }

  GenericFunc::Get(func_name)
    .register_func(tags_vector, *func, allow_override);
  });

TVM_REGISTER_API("_InvokeGenericFunc")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string func_name = args[0];
  std::string target_str = args[1];

  TVMArgs func_args(&args.values[2], &args.type_codes[2], args.num_args - 2);

  auto target = Target::create(target_str);
  GenericFunc::Get(func_name)
    .invoke_func(target, func_args, ret);
  });

}  // namespace tvm

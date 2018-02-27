/*!
 *  Copyright (c) 2017 by Contributors
 *  Compile executable modules.
 * \file build_module.cc
 */
#include <tvm/build_module.h>
#include <tvm/operation.h>
#include <tvm/ir_pass.h>
#include <tvm/codegen.h>


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
    return target::rocm();
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
  std::unordered_set<std::string> keys({ "llvm", "cpu" });
  std::vector<std::string> options;
  return Target("llvm", kDLCPU, 512, 1, keys, options,
           std::unordered_set<std::string>());
}

Target cuda() {
  std::unordered_set<std::string> keys({ "cuda", "gpu" });
  std::vector<std::string> options;
  return Target("cuda", kDLGPU, 512, 32, keys, options,
           std::unordered_set<std::string>());
}

Target rocm() {
  std::unordered_set<std::string> keys({ "rocm", "gpu" });
  std::vector<std::string> options;
  return Target("rocm", kDLROCM, 256, 1, keys, options,
           std::unordered_set<std::string>());
}

Target metal() {
  std::unordered_set<std::string> keys({ "gpu" });
  std::vector<std::string> options;
  return Target("metal", kDLMetal, 256, 1, keys, options,
           std::unordered_set<std::string>());
}

Target rasp() {
  std::unordered_set<std::string> keys({ "llvm", "cpu" });
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
  std::unordered_set<std::string> keys({ "rocm", "gpu" });
  std::vector<std::string> options({
    "-device=mali"
  });
  return Target("opencl", kDLOpenCL, 256, 1, keys, options);
}


Target stackvm() {
  std::unordered_set<std::string> keys({ "stackvm", "cpu" });
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

  if (target.keys.count("gpu") > 0 && fdevice.size() == 0) {
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

}  // namespace tvm

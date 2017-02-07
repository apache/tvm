/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_cuda.cc
 */
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "./codegen_opencl.h"
#include "./codegen_stack_vm.h"
#include "../runtime/opencl/opencl_common.h"
#include "../runtime/opencl/opencl_module.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace codegen {

std::string CodeGenOpenCL::Compile(
    LoweredFunc f,
    bool output_ssa) {
  this->stream << " __kernel ";
  this->arg_addr_space_ = "__global ";
  return CodeGenC::Compile(f, output_ssa);
}

void CodeGenOpenCL::PrintThreadIndexExpr(
    std::string tag, std::ostream& os) { // NOLINT(*)
  runtime::ThreadScope ts = runtime::ThreadScope::make(tag);
  if (ts.rank == 1) {
    os << "get_local_id(" << ts.dim_index << ")";
  } else {
    os << "get_global_id(" << ts.dim_index << ")"
       << " / get_local_size(" << ts.dim_index << ")";
  }
}


void CodeGenOpenCL::PrintStorageSync(const std::string& sync) {
  if (sync == "shared") {
    this->PrintIndent();
    this->stream << "barrier(CLK_LOCAL_MEM_FENCE);\n";
  } else if (sync == "global") {
    LOG(FATAL) << "not supported";
  }
}

void CodeGenOpenCL::PrintStorageScope(const std::string& scope, std::ostream& os) { // NOLINT(*)
  CHECK_NE(scope, "global");
  if (scope == "shared") {
    os << "__local ";
  }
}

#if TVM_OPENCL_RUNTIME
std::unordered_map<LoweredFunc, PackedFunc>
MakeOpenCL(Array<LoweredFunc> funcs) {
  std::ostringstream os;
  os << "typedef int int32_t;\n"
     << "typedef unsigned unt32_t;\n";
  bool output_ssa = false;
  for (LoweredFunc f : funcs) {
    os << CodeGenOpenCL().Compile(f, output_ssa);
    os << '\n';
  }
  std::unordered_map<LoweredFunc, PackedFunc> ret;
  runtime::OpenCLModule m =
      runtime::OpenCLModule::CreateWithSource(os.str());
  for (LoweredFunc f : funcs) {
    std::vector<TVMType> arg_types(f->args.size());
    std::vector<std::string> thread_axis_tags(f->thread_axis.size());

    for (size_t i = 0; i < f->args.size(); ++i) {
      arg_types[i] = Type2TVMType(f->args[i].type());
    }
    for (size_t i = 0; i < f->thread_axis.size(); ++i) {
      thread_axis_tags[i] = f->thread_axis[i]->thread_tag;
    }
    ret[f] = m.GetPackedFunc(f->name, arg_types, thread_axis_tags);
  }
  return ret;
}

PackedFunc BuildOpenCL(Array<LoweredFunc> fsplits, std::string host_mode) {
  Array<LoweredFunc> device_list(fsplits.begin() + 1, fsplits.end());
  std::unordered_map<LoweredFunc, PackedFunc> device_funcs = MakeOpenCL(device_list);

  if (host_mode == "stackvm") {
    StackVM vm = codegen::CodeGenStackVM().Compile(fsplits[0], device_funcs);
    auto f = [vm](TVMArgs args, TVMRetValue* rv) {
      runtime::AutoSetOpenCLContext(args);
      vm(args);
    };
    return PackedFunc(f);
  } else {
    LOG(FATAL) << "unknown host mode " << host_mode;
    return PackedFunc();
  }
}
#else
// dummy function when opencl is not available
PackedFunc BuildOpenCL(Array<LoweredFunc> func, std::string host_mode) {
  LOG(FATAL) << "OpenCL is not enabled";
  return PackedFunc();
}
#endif   // TVM_OPENCL_RUNTIME
}  // namespace codegen
}  // namespace tvm

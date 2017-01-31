/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_cuda.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "./codegen_cuda.h"
#include "./codegen_stack_vm.h"
#include "../runtime/cuda/cuda_common.h"
#include "../runtime/cuda/cuda_module.h"

namespace tvm {
namespace codegen {

std::string CodeGenCUDA::Compile(
    LoweredFunc f,
    bool output_ssa) {
  this->stream << "extern \"C\" __global__ ";
  return CodeGenC::Compile(f, output_ssa);
}

#if TVM_CUDA_RUNTIME
std::unordered_map<LoweredFunc, PackedFunc>
MakeNVRTC(Array<LoweredFunc> funcs) {
  std::ostringstream os;
  os << "typedef int int32_t;\n"
     << "typedef unsigned unt32_t;\n";
  bool output_ssa = true;
  for (LoweredFunc f : funcs) {
    os << CodeGenCUDA().Compile(f, output_ssa);
    os << '\n';
  }
  std::string ptx = runtime::NVRTCCompile(os.str());
  std::unordered_map<LoweredFunc, PackedFunc> ret;

  runtime::CUDAModule m = runtime::CUDAModule::Create(ptx);
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

PackedFunc BuildNVRTC(Array<LoweredFunc> fsplits, std::string host_mode) {
  Array<LoweredFunc> device_list(fsplits.begin() + 1, fsplits.end());
  std::unordered_map<LoweredFunc, PackedFunc> device_funcs = MakeNVRTC(device_list);

  if (host_mode == "stackvm") {
    StackVM vm = codegen::CodeGenStackVM().Compile(fsplits[0], device_funcs);
    auto f = [vm](TVMArgs args, TVMRetValue* rv) {
      runtime::AutoSetCUDADevice(args);
      vm(args);
    };
    return PackedFunc(f);
  } else {
    LOG(FATAL) << "unknown host mode " << host_mode;
    return PackedFunc();
  }
}
#else
// dummy function when cuda is not available
PackedFunc BuildNVRTC(Array<LoweredFunc> func, std::string host_mode) {
  LOG(FATAL) << "CUDA is not enabled";
  return PackedFunc();
}
#endif   // TVM_CUDA_RUNTIME
}  // namespace codegen
}  // namespace tvm

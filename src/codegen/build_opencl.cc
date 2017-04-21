/*!
 *  Copyright (c) 2017 by Contributors
 *  Build opencl modules from source.
 * \file build_opencl.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include "./codegen_opencl.h"

#if TVM_OPENCL_RUNTIME

#include "../runtime/meta_data.h"
#include "../runtime/opencl/opencl_common.h"
#include "../runtime/opencl/opencl_module.h"

namespace tvm {
namespace codegen {

runtime::Module BuildOpenCL(Array<LoweredFunc> funcs) {
  std::ostringstream os;
  bool output_ssa = false;
  CodeGenOpenCL cg;
  cg.Init(output_ssa);

  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();

  std::unordered_map<std::string, runtime::FunctionInfo> fmap;
  for (LoweredFunc f : funcs) {
    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->args.size(); ++i) {
      info.arg_types.push_back(Type2TVMType(f->args[i].type()));
    }
    for (size_t i = 0; i < f->thread_axis.size(); ++i) {
      info.thread_axis_tags.push_back(f->thread_axis[i]->thread_tag);
    }
    fmap[f->name] = info;
  }
  return OpenCLModuleCreate(code, "cl", fmap);
}

TVM_REGISTER_API("codegen.build_opencl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildOpenCL(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
#endif   // TVM_OPENCL_RUNTIME

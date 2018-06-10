/*!
 *  Copyright (c) 2017 by Contributors
 *  Build opencl modules from source.
 * \file build_opencl.h
 */
#ifndef TVM_CODEGEN_OPT_BUILD_OPENCL_H_
#define TVM_CODEGEN_OPT_BUILD_OPENCL_H_

#include <tvm/base.h>
#include <string>
#include "../codegen_opencl.h"
#include "../build_common.h"

#if TVM_OPENCL_RUNTIME
#include "../../runtime/opencl/opencl_module.h"
#endif   // TVM_OPENCL_RUNTIME

namespace tvm {
namespace codegen {

runtime::Module BuildOpenCL(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenOpenCL cg;
  cg.Init(output_ssa);
  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_opencl_postproc")) {
    code = (*f)(code).operator std::string();
  }
#if TVM_OPENCL_RUNTIME
  return OpenCLModuleCreate(code, "cl", ExtractFuncInfo(funcs));
#else
  LOG(WARNING) << "OpenCL runtime not enabled, return a source module...";
  return DeviceSourceModuleCreate(code, "cl", ExtractFuncInfo(funcs), "opencl");
#endif   // TVM_OPENCL_RUNTIME
}

TVM_REGISTER_API("codegen.build_opencl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildOpenCL(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_OPT_BUILD_OPENCL_H_

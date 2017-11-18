/*!
 *  Copyright (c) 2017 by Contributors
 *  Build metal modules from source.
 * \file build_metal.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include "./codegen_metal.h"
#include "./build_common.h"

#if TVM_METAL_RUNTIME
#include "../runtime/metal/metal_module.h"
#endif   // TVM_METAL_RUNTIME

namespace tvm {
namespace codegen {

runtime::Module BuildMetal(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenMetal cg;
  cg.Init(output_ssa);
  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();
#if TVM_METAL_RUNTIME
  std::string fmt = "metal";
  std::string source = "";
  if (const auto* f = Registry::Get("tvm_callback_metal_compile")) {
    source = code;
    code = (*f)(code).operator std::string();
    fmt = "metallib";
  }
  return MetalModuleCreate(code, fmt, ExtractFuncInfo(funcs), source);
#else
  LOG(WARNING) << "Metal runtime not enabled, return a source module...";
  return DeviceSourceModuleCreate(code, "metal", ExtractFuncInfo(funcs), "metal");
#endif   // TVM_METAL_RUNTIME
}

TVM_REGISTER_API("codegen.build_metal")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildMetal(args[0]);
  });
}  // namespace codegen
}  // namespace tvm

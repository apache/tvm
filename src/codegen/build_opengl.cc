/*!
 *  Copyright (c) 2017 by Contributors
 *  Build opengl modules from source.
 * \file build_opengl.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include "./codegen_opengl.h"
#include "./build_common.h"

#if TVM_OPENGL_RUNTIME
#include "../runtime/opengl/opengl_module.h"
#endif   // TVM_OPENGL_RUNTIME

namespace tvm {
namespace codegen {

runtime::Module BuildOpenGL(Array<LoweredFunc> funcs) {
  bool output_ssa = false;
  CodeGenOpenGL cg;
  cg.Init(output_ssa);
  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();
#if TVM_OPENCL_RUNTIME
  return OpenGLModuleCreate(code, "gl", ExtractFuncInfo(funcs));
#else
  LOG(WARNING) << "OpenCL runtime not enabled, return a source module...";
  return DeviceSourceModuleCreate(code, "cl", ExtractFuncInfo(funcs), "opencl");
#endif   // TVM_OPENGL_RUNTIME
}

TVM_REGISTER_API("codegen.build_opengl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = BuildOpenGL(args[0]);
});
}  // namespace codegen
}  // namespace tvm

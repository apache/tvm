/*!
 *  Copyright (c) 2018 by Contributors
 *  Optional module when build opencl is switched to off
 */
#include "../codegen_source_base.h"
#include "../../runtime/opengl/opengl_module.h"

namespace tvm {
namespace runtime {

Module OpenGLModuleCreate(std::unordered_map<std::string, OpenGLShader> shaders,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap) {
  LOG(WARNING) << "OpenGL runtime not enabled, return a source module...";
  auto data = ToJSON(shaders);
  return codegen::DeviceSourceModuleCreate(data, "gl", fmap, "opengl");
}

}  // namespace runtime
}  // namespace tvm

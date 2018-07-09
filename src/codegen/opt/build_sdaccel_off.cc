/*!
 *  Copyright (c) 2018 by Contributors
 *  Optional module when build opencl is switched to off
 */
#include "../codegen_source_base.h"
#include "../../runtime/opencl/opencl_module.h"

namespace tvm {
namespace runtime {

Module SDAccelModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source) {
  LOG(WARNING) << "OpenCL runtime not enabled, return a source module...";
  return codegen::DeviceSourceModuleCreate(data, fmt, fmap, "sdaccel");
}

}  // namespace runtime
}  // namespace tvm

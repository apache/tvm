/*!
 *  Copyright (c) 2018 by Contributors
 *  Optional module when build aocl is switched to off
 */
#include "../codegen_source_base.h"
#include "../../runtime/opencl/opencl_module.h"

namespace tvm {
namespace runtime {

Module AOCLModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source) {
  LOG(WARNING) << "AOCL runtime not enabled, return a source module...";
  return codegen::DeviceSourceModuleCreate(data, fmt, fmap, "aocl");
}

}  // namespace runtime
}  // namespace tvm

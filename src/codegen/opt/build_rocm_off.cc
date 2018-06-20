/*!
 *  Copyright (c) 2018 by Contributors
 *  Optional module when build rocm is switched to off
 */
#include "../codegen_source_base.h"
#include "../../runtime/rocm/rocm_module.h"

namespace tvm {
namespace runtime {

Module ROCMModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string rocm_source,
    std::string assembly) {

  LOG(WARNING) << "ROCM runtime is not enabled, return a source module...";
  auto fget_source = [rocm_source, assembly](const std::string& format) {
    if (format.length() == 0) return assembly;
    if (format == "ll" || format == "llvm") return rocm_source;
    if (format == "asm") return assembly;
    return std::string("");
  };
  return codegen::DeviceSourceModuleCreate(
      data, fmt, fmap, "hsaco", fget_source);
}

}  // namespace runtime
}  // namespace tvm

/*!
 *  Copyright (c) 2018 by Contributors
 *  Optional module when build metal is switched to off
 */
#include "../codegen_source_base.h"
#include "../../runtime/metal/metal_module.h"

namespace tvm {
namespace runtime {

Module MetalModuleCreate(std::string data,
                         std::string fmt,
                         std::unordered_map<std::string, FunctionInfo> fmap,
                         std::string source) {
  LOG(WARNING) << "Metal runtime not enabled, return a source module...";
  return codegen::DeviceSourceModuleCreate(data, fmt, fmap, "metal");
}

}  // namespace runtime
}  // namespace tvm

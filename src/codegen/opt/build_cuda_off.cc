/*!
 *  Copyright (c) 2018 by Contributors
 *  Optional module when build cuda is switched to off
 */
#include "../../runtime/cuda/cuda_module.h"
namespace tvm {
namespace runtime {

Module CUDAModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string cuda_source) {
  LOG(FATAL) << "CUDA is not enabled";
  return Module();
}
}  // namespace runtime
}  // namespace tvm

/*!
 *  Copyright (c) 2017 by Contributors
 * \file nvrtc.cc
 */
#include "./cuda_common.h"

#if TVM_CUDA_RUNTIME

#include <nvrtc.h>

namespace tvm {
namespace runtime {

#define NVRTC_CALL(x)                                                   \
  {                                                                     \
    nvrtcResult result = x;                                             \
    if (result != NVRTC_SUCCESS) {                                      \
      LOG(FATAL)                                                        \
          << "NvrtcError: " #x " failed with error: "                   \
          << nvrtcGetErrorString(result);                               \
    }                                                                   \
  }

std::string NVRTCCompile(const std::string& code) {
  nvrtcProgram prog;
  NVRTC_CALL(nvrtcCreateProgram(
      &prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  nvrtcResult compile_res = nvrtcCompileProgram(prog, 0, nullptr);
  size_t log_size;
  NVRTC_CALL(nvrtcGetProgramLogSize(prog, &log_size));
  std::string log; log.resize(log_size);
  NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));
  CHECK_EQ(compile_res, NVRTC_SUCCESS) << log;
  size_t ptx_size;
  NVRTC_CALL(nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  NVRTC_CALL(nvrtcGetPTX(prog, &ptx[0]));
  NVRTC_CALL(nvrtcDestroyProgram(&prog));
  return ptx;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_CUDA_RUNTIME

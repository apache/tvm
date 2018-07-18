/*!
 *  Copyright (c) 2017 by Contributors
 *  Build cuda modules from source.
 *  requires cuda to be available.
 *
 * \file build_cuda.cc
 */
#include <cuda_runtime.h>
#include <tvm/base.h>
#include <nvrtc.h>
#include <cstdlib>

#include "../codegen_cuda.h"
#include "../build_common.h"
#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/cuda/cuda_module.h"

#ifdef _WIN32
#define PATH_DELIMITER "\\"
#else
#define PATH_DELIMITER "/"
#endif

namespace tvm {
namespace codegen {

#define NVRTC_CALL(x)                                                   \
  {                                                                     \
    nvrtcResult result = x;                                             \
    if (result != NVRTC_SUCCESS) {                                      \
      LOG(FATAL)                                                        \
          << "NvrtcError: " #x " failed with error: "                   \
          << nvrtcGetErrorString(result);                               \
    }                                                                   \
  }

std::string NVRTCCompile(const std::string& code, bool include_path = false) {
  std::vector<std::string> compile_params;
  std::vector<const char*> param_cstrings{};
  int num_options = 0;
  nvrtcProgram prog;
  cudaDeviceProp device_prop;
  std::string cc = "30";
  cudaError_t e = cudaGetDeviceProperties(&device_prop, 0);

  if (e == cudaSuccess) {
    cc = std::to_string(device_prop.major) + std::to_string(device_prop.minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
  }

  compile_params.push_back("-arch=compute_" + cc);
  num_options++;

  if (include_path) {
    std::string include_option = "--include-path=";
    const char* cudaHomePath = std::getenv("CUDA_HOME");

    if (cudaHomePath != nullptr) {
      include_option += cudaHomePath;
      include_option += PATH_DELIMITER;
      include_option += "include";
    } else {
      LOG(FATAL)
          << "NvrtcError: Set the environment variables CUDA_HOME to the location of cuda";
    }

    compile_params.push_back(include_option);
    num_options++;
  }

  for (const auto& string : compile_params)
      param_cstrings.push_back(string.c_str());
  NVRTC_CALL(nvrtcCreateProgram(
      &prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  nvrtcResult compile_res = nvrtcCompileProgram(prog, num_options, param_cstrings.data());

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

runtime::Module BuildCUDA(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenCUDA cg;
  cg.Init(output_ssa);

  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_cuda_postproc")) {
    code = (*f)(code).operator std::string();
  }
  std::string fmt = "ptx";
  std::string ptx;
  if (const auto* f = Registry::Get("tvm_callback_cuda_compile")) {
    ptx = (*f)(code).operator std::string();
    // Dirty matching to check PTX vs cubin.
    // TODO(tqchen) more reliable checks
    if (ptx[0] != '/') fmt = "cubin";
  } else {
    ptx = NVRTCCompile(code, cg.need_include_path());
  }
  return CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(funcs), code);
}

TVM_REGISTER_API("codegen.build_cuda")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildCUDA(args[0]);
  });
}  // namespace codegen
}  // namespace tvm

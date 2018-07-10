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
  char *compileParams[2];
  int numCompileOptions = 0;
  nvrtcProgram prog;
  cudaDeviceProp deviceProp;
  std::string cc = "30";
  cudaError_t e = cudaGetDeviceProperties(&deviceProp, 0);

  if (e == cudaSuccess) {
    cc = std::to_string(deviceProp.major) + std::to_string(deviceProp.minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
  }

  std::string archOption = "-arch=compute_" + cc;
  compileParams[numCompileOptions] = reinterpret_cast<char *>(malloc(sizeof(char) *
                                                                     (archOption.length() + 1)));
  snprintf(compileParams[numCompileOptions], archOption.length() + 1, "%s", archOption.c_str());
  numCompileOptions++;

  if (include_path) {
    std::string includeOption = "--include-path=";
    const char* cudaHomePath = std::getenv("CUDA_HOME");

    if (cudaHomePath != nullptr) {
      includeOption += cudaHomePath;
      includeOption += "/include";
    } else {
      LOG(FATAL)
          << "NvrtcError: Set the environment variables CUDA_HOME to the location of cuda";
    }

    compileParams[numCompileOptions] = reinterpret_cast<char *>(malloc(sizeof(char) *
                                                       (includeOption.length() + 1)));
    snprintf(compileParams[numCompileOptions], includeOption.length() + 1, "%s",
             includeOption.c_str());
    numCompileOptions++;
  }

  NVRTC_CALL(nvrtcCreateProgram(
      &prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  nvrtcResult compile_res = nvrtcCompileProgram(prog, numCompileOptions, compileParams);

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

  if (include_path) {
    for (int i = 0; i < numCompileOptions; i++) {
      free(compileParams[i]);
    }
  }

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

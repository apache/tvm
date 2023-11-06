/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Build cuda modules from source.
 *  requires cuda to be available.
 *
 * \file build_cuda.cc
 */
#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdlib>

#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/cuda/cuda_module.h"
#include "../build_common.h"
#include "../source/codegen_cuda.h"

namespace tvm {
namespace codegen {

#define NVRTC_CALL(x)                                                                        \
  {                                                                                          \
    nvrtcResult result = x;                                                                  \
    if (result != NVRTC_SUCCESS) {                                                           \
      LOG(FATAL) << "NvrtcError: " #x " failed with error: " << nvrtcGetErrorString(result); \
    }                                                                                        \
  }

std::string FindCUDAIncludePath() {
#if defined(_WIN32)
  const std::string delimiter = "\\";
#else
  const std::string delimiter = "/";
#endif
  std::string cuda_include_path;
  const char* cuda_path_env = std::getenv("CUDA_PATH");
  if (cuda_path_env != nullptr) {
    cuda_include_path += cuda_path_env;
    cuda_include_path += delimiter + "include";
    return cuda_include_path;
  }

#if defined(__linux__)
  struct stat st;
  cuda_include_path = "/usr/local/cuda/include";
  if (stat(cuda_include_path.c_str(), &st) == 0) {
    return cuda_include_path;
  }

  if (stat("/usr/include/cuda.h", &st) == 0) {
    return "/usr/include";
  }
#endif
  LOG(FATAL) << "Cannot find cuda include path."
             << "CUDA_PATH is not set or CUDA is not installed in the default installation path."
             << "In other than linux, it is necessary to set CUDA_PATH.";
  return cuda_include_path;
}

std::string NVRTCCompile(const std::string& code, bool include_path = false) {
  std::vector<std::string> compile_params;
  std::vector<const char*> param_cstrings{};
  nvrtcProgram prog;
  std::string cc = "30";
  int major, minor;
  cudaError_t e1 = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaError_t e2 = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

  if (e1 == cudaSuccess && e2 == cudaSuccess) {
    cc = std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
  }

  compile_params.push_back("-arch=compute_" + cc);

  if (include_path) {
    std::string include_option = "--include-path=" + FindCUDAIncludePath();

    compile_params.push_back(include_option);
  }

  for (const auto& string : compile_params) {
    param_cstrings.push_back(string.c_str());
  }
  NVRTC_CALL(nvrtcCreateProgram(&prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  nvrtcResult compile_res = nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  size_t log_size;
  NVRTC_CALL(nvrtcGetProgramLogSize(prog, &log_size));
  std::string log;
  log.resize(log_size);
  NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));
  ICHECK_EQ(compile_res, NVRTC_SUCCESS) << log;
  size_t ptx_size;
  NVRTC_CALL(nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  NVRTC_CALL(nvrtcGetPTX(prog, &ptx[0]));
  NVRTC_CALL(nvrtcDestroyProgram(&prog));

  return ptx;
}

runtime::Module BuildCUDA(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenCUDA cg;
  cg.Init(output_ssa);

  Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenCUDA: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    auto calling_conv = prim_func->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenCUDA: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    functions.Set(gvar, prim_func);
  }

  for (auto [gvar, prim_func] : functions) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (auto [gvar, prim_func] : functions) {
    cg.AddFunction(gvar, prim_func);
  }

  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_cuda_postproc")) {
    code = (*f)(code, target).operator std::string();
  }
  std::string fmt = "ptx";
  std::string ptx;
  const auto* f_enter = Registry::Get("target.TargetEnterScope");
  (*f_enter)(target);
  if (const auto* f = Registry::Get("tvm_callback_cuda_compile")) {
    ptx = (*f)(code, target).operator std::string();
    // Dirty matching to check PTX vs cubin.
    // TODO(tqchen) more reliable checks
    if (ptx[0] != '/') fmt = "cubin";
  } else {
    ptx = NVRTCCompile(code, cg.need_include_path());
  }
  const auto* f_exit = Registry::Get("target.TargetExitScope");
  (*f_exit)(target);
  return CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.cuda").set_body_typed(BuildCUDA);
TVM_REGISTER_PASS_CONFIG_OPTION("cuda.kernels_output_dir", String);
}  // namespace codegen
}  // namespace tvm

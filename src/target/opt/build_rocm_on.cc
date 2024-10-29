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
 *  Optional module when build rocm is switched to on
 */

#if defined(__linux__)
#include <sys/stat.h>
#endif

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <string>
#include <vector>

#include "../../runtime/rocm/rocm_module.h"
#include "../build_common.h"
#include "../source/codegen_hip.h"
#include "../source/codegen_source_base.h"

namespace tvm {
namespace codegen {

#define HIPRTC_CALL(x)                                                                  \
  \  
  {                                                                                     \
    \  
    hiprtcResult result = x;                                                            \
    \  
    if (result != HIPRTC_SUCCESS) {                                                     \
      \  
      LOG(FATAL)                                                                        \
          << "HiprtcError: " #x " failed with error: " << hiprtcGetErrorString(result); \
      \  
                                                                                   \
    }                                                                                   \
    \  
                                                                                   \
  }

std::string FindHIPIncludePath() {
#if defined(_WIN32)
  const std::string delimiter = "\\";
#else
  const std::string delimiter = "/";
#endif
  std::string hip_include_path;
  const char* hip_path_env = std::getenv("HIP_PATH");
  if (hip_path_env != nullptr) {
    hip_include_path += hip_path_env;
    hip_include_path += delimiter + "include";
    return hip_include_path;
  }

#if defined(__linux__)
  struct stat st;
  hip_include_path = "/opt/rocm/hip/include";
  if (stat(hip_include_path.c_str(), &st) == 0) {
    return hip_include_path;
  }

  if (stat("/usr/include/hip/hip_runtime.h", &st) == 0) {
    return "/usr/include/hip";
  }
#endif
  LOG(FATAL) << "Cannot find HIP include path."
             << "HIP_PATH is not set or ROCm is not installed in the default installation path."
             << "In other than linux, it is necessary to set HIP_PATH.";
  return hip_include_path;
}

std::string HIPRTCCompile(const std::string& code, bool include_path = false) {
  std::vector<std::string> compile_params;
  std::vector<const char*> param_cstrings{};
  hiprtcProgram prog;
  std::string cc = "gfx900";  // Default target architecture (can be changed as needed)
  int major, minor;
  hipError_t e1 = hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, 0);
  hipError_t e2 = hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, 0);

  if (e1 == hipSuccess && e2 == hipSuccess) {
    cc = "gfx" + std::to_string(major * 100 + minor * 10);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to gfx900.";
  }

  compile_params.push_back("--gpu-architecture=" + cc);

  if (include_path) {
    std::string include_option = "--include-path=" + FindHIPIncludePath();
    compile_params.push_back(include_option);
  }

  for (const auto& string : compile_params) {
    param_cstrings.push_back(string.c_str());
  }
  HIPRTC_CALL(hiprtcCreateProgram(&prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  hiprtcResult compile_res =
      hiprtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  size_t log_size;
  HIPRTC_CALL(hiprtcGetProgramLogSize(prog, &log_size));
  std::string log;
  log.resize(log_size);
  HIPRTC_CALL(hiprtcGetProgramLog(prog, &log[0]));
  ICHECK_EQ(compile_res, HIPRTC_SUCCESS) << log;
  size_t code_size;
  HIPRTC_CALL(hiprtcGetCodeSize(prog, &code_size));

  std::string code_out;
  code_out.resize(code_size);
  HIPRTC_CALL(hiprtcGetCode(prog, &code_out[0]));
  HIPRTC_CALL(hiprtcDestroyProgram(&prog));

  return code_out;
}

runtime::Module BuildHIP(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenHIP cg;
  cg.Init(output_ssa);

  Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenHIP: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    auto calling_conv = prim_func->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenHIP: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    functions.Set(gvar, prim_func);
  }

  for (auto [gvar, prim_func] : functions) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (auto [gvar, prim_func] : functions) {
    cg.AddFunction(gvar, prim_func);
  }
  
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_hip_postproc")) {
    code = (*f)(code, target).operator std::string();
  }
  std::string fmt = "ptx";
  std::string ptx;
  const auto* f_enter = Registry::Get("target.TargetEnterScope");
  (*f_enter)(target);
  if (const auto* f = Registry::Get("tvm_callback_hip_compile")) {
    ptx = (*f)(code, target).operator std::string();
    // Dirty matching to check PTX vs hsaco.
    // TODO(leiwang1999) more reliable checks
    if (ptx[0] != '/') fmt = "hsaco";
  } else {
    ptx = HIPRTCCompile(code, cg.need_include_path());
  }
  const auto* f_exit = Registry::Get("target.TargetExitScope");
  (*f_exit)(target);
  return ROCMModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code, std::string());
}

TVM_REGISTER_GLOBAL("target.build.hip").set_body_typed(BuildHIP);
}  // namespace codegen
}  // namespace tvm

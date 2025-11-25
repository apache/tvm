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
#include <tvm/ffi/reflection/registry.h>
#endif
#include <cuda_runtime.h>

#include <cstdlib>

#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/cuda/cuda_module.h"
#include "../build_common.h"
#include "../source/codegen_cuda.h"

namespace tvm {
namespace codegen {

// Note: CUDA include path finding and NVRTC compilation are now handled
// in Python (tvm.contrib.nvrtc) for better maintainability and to leverage
// cuda-python bindings. The C++ NVRTC code has been removed as part of the
// Python-first compilation strategy.

ffi::Module BuildCUDA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenCUDA cg;
  cg.Init(output_ssa);

  ffi::Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenCUDA: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    auto calling_conv =
        prim_func->GetAttr<Integer>(tvm::attr::kCallingConv, Integer(tvm::CallingConv::kDefault));
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch ||
           calling_conv == CallingConv::kDefault)
        << "CodeGenCUDA: expect calling_conv equals CallingConv::kDeviceKernelLaunch or "
           "CallingConv::kDefault";
    functions.Set(gvar, prim_func);
  }

  for (auto [gvar, prim_func] : functions) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (auto [gvar, prim_func] : functions) {
    cg.AddFunction(gvar, prim_func);
  }

  std::string code = cg.Finish();

  // Apply optional post-processing callback
  if (auto f = ffi::Function::GetGlobal("tvm_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  // Always use Python compilation callback (nvcc or nvrtc)
  // The C++ NVRTC fallback has been removed in favor of Python-first approach
  auto f_compile = ffi::Function::GetGlobal("tvm_callback_cuda_compile");
  ICHECK(f_compile != nullptr)
      << "tvm_callback_cuda_compile not found. "
      << "Please ensure TVM Python runtime is properly initialized.\n"
      << "The Python callback (tvm.contrib.nvcc.tvm_callback_cuda_compile) is required "
      << "for CUDA compilation. The C++ NVRTC fallback has been removed.\n"
      << "Make sure to import tvm.contrib.nvcc in your Python code.";

  // Enter target scope for compilation
  auto f_enter = ffi::Function::GetGlobal("target.TargetEnterScope");
  (*f_enter)(target);

  // Compile CUDA code via Python callback
  std::string compiled = (*f_compile)(code, target).cast<std::string>();

  // Exit target scope
  auto f_exit = ffi::Function::GetGlobal("target.TargetExitScope");
  (*f_exit)(target);

  // Auto-detect format: PTX starts with '//' or '.version', CUBIN is binary
  std::string fmt = "cubin";
  if (!compiled.empty()) {
    // PTX format detection
    if (compiled[0] == '/' || compiled[0] == '.') {
      fmt = "ptx";
    }
    // FATBIN format detection (starts with specific magic bytes)
    // FATBIN magic: 0x00, 0x00, 0xBA, 0x00 or similar
    // For simplicity, if it's not PTX, assume it's binary (cubin or fatbin)
    // The CUDAModuleCreate will handle both correctly
  }

  return CUDAModuleCreate(compiled, fmt, ExtractFuncInfo(mod), code);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.cuda", BuildCUDA);
}
TVM_REGISTER_PASS_CONFIG_OPTION("cuda.kernels_output_dir", ffi::String);
}  // namespace codegen
}  // namespace tvm

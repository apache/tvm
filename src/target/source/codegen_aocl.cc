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
 * \file codegen_aocl.cc
 */
#include <tvm/target/target.h>
#include <vector>
#include <string>
#include "codegen_opencl.h"
#include "../build_common.h"
#include "../../runtime/opencl/aocl/aocl_module.h"
#include "../../runtime/file_util.h"

namespace tvm {
namespace codegen {

runtime::Module BuildAOCL(IRModule mod,
                          std::string target_str,
                          bool emulation) {
  // Get code.
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenOpenCL cg;
  cg.Init(output_ssa);

  for (auto kv :  mod->functions) {
    CHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodegenOpenCL: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    CHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodegenOpenCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();
  if (const auto* f = Registry::Get("tvm_callback_opencl_postproc")) {
    code = (*f)(code).operator std::string();
  }

  // Write a .cl file.
  runtime::SaveBinaryToFile("aocl.cl", code.c_str());

  // Compile the .cl file.
  std::string cmd = "aoc aocl.cl";
  // AOCL supports fp64.
  cmd += " -Dcl_khr_fp64";
  Target target = Target::Create(target_str);
  if (target->device_name != "") {
    cmd += " -board=" + target->device_name;
  }
  if (emulation) {
    cmd += " -march=emulator";
  }
  if (system(cmd.c_str()) != 0) {
    LOG(FATAL) << "OpenCL offline compilation error.";
  }

  // Read .aocx file
  std::string aocxbin;
  runtime::LoadBinaryFromFile("aocl.aocx", &aocxbin);

  return AOCLModuleCreate(aocxbin, "aocx", ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.aocl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildAOCL(args[0], args[1], false);
  });

TVM_REGISTER_GLOBAL("target.build.build.aocl_sw_emu")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildAOCL(args[0], args[1], true);
  });

}  // namespace codegen
}  // namespace tvm

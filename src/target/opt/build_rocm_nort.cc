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
 *  Optional module when build rocm is switched to off
 */

#include <string>
#include <vector>

#include "../../runtime/rocm/rocm_module.h"
#include "../build_common.h"
#include "../source/codegen_hip.h"
#include "../source/codegen_source_base.h"

namespace tvm {
namespace runtime {

class ROCMModuleNode : public runtime::ModuleNode {
 public:
  explicit ROCMModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string hip_source, std::string assembly)
      : data_(data), fmt_(fmt), fmap_(fmap), hip_source_(hip_source), assembly_(assembly) {}

  const char* type_key() const final { return "hip"; }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
      ICHECK(0) << "Not implemented when rocm is not enabled in TVM.";
      return PackedFunc();
  }


  String GetSource(const String& format) {
    if (format == fmt_) {
      return data_;
    }
    if (format == "llvm" || format == "") {
      return hip_source_;
    }
    if (format == "asm") {
      return assembly_;
    }
    return "";
  }


 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The hip source.
  std::string hip_source_;
  // The gcn asm.
  std::string assembly_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

Module ROCMModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap, std::string hip_source,
                        std::string assembly) {
  auto n = make_object<tvm::runtime::ROCMModuleNode>(data, fmt, fmap, hip_source, assembly);
  return Module(n);
}

}  // namespace runtime
}  // namespace tvm
namespace tvm {
namespace codegen {
using tvm::runtime::Registry;
runtime::Module BuildHIP(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenHIP cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenHIP: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenHIP: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_hip_postproc")) {
    code = (*f)(code, target).operator std::string();
  }
  std::string fmt = "ptx";
  std::string ptx;
  const auto* f_enter = Registry::Get("target.TargetEnterScope");
  (*f_enter)(target);
  const auto* f_exit = Registry::Get("target.TargetExitScope");
  (*f_exit)(target);
  return ROCMModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code, std::string());
}

TVM_REGISTER_GLOBAL("target.build.hip").set_body_typed(BuildHIP);
}  // namespace codegen
}  // namespace tvm

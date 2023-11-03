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


#include "../runtime/cuda/cuda_module.h"
#include "../target/build_common.h"
#include "codegen.h"

namespace tvm {
namespace codegen {

runtime::Module BuildTLDebug(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenTL cg;
  cg.Init(output_ssa);
  ICHECK(mod->functions.size() == 1) << "Currently support one kernel.";

  auto kv = *mod->functions.begin();
  ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenTL: Can only take PrimFunc";
  auto f = Downcast<PrimFunc>(kv.second);
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  cg.AddFunction(f);
  String code = cg.Finish();

  std::string fmt = "ptx";
  std::string ptx;
  if (const auto* f = Registry::Get("tvm_tl_cuda_compile")) {
    ptx = (*f)(code, target).operator std::string();
    // Dirty matching to check PTX vs cubin.
    // TODO(tqchen) more reliable checks
    if (ptx[0] != '/') fmt = "cubin";
  } else {
    ICHECK(0);
  }
  return runtime::CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.tl").set_body_typed(BuildTLDebug);

}  // namespace codegen
}  // namespace tvm

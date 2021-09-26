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
#include <tvm/relay/transform.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

transform::Pass RelayToTIR();

runtime::Module CompileCMSISNN(const ObjectRef& ref) {
  IRModule relay_mod;
  Function relay_func = Downcast<Function>(ref);
  auto func_name = relay_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  GlobalVar var = GlobalVar(func_name.value());
  relay_mod->Add(var, relay_func);
  relay_mod = transform::InferType()(relay_mod);

  Array<transform::Pass> pass_seqs{transform::InferType(), RelayToTIR()};
  transform::Sequential seq(pass_seqs);
  IRModule tir_mod = seq(relay_mod);

  const auto* pf = runtime::Registry::Get("runtime.CMSISNNModuleNodeCreate");
  return (*pf)(tir_mod);
}

TVM_REGISTER_GLOBAL("relay.ext.cmsisnn").set_body_typed(CompileCMSISNN);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

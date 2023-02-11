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
 * \file src/relax/transform/attach_global_symbol.cc
 * \brief Attach global_symbol to Relax functions and TIR Primfuncs for codegen.
 */

#include <tvm/relax/transform.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace relax {

class GlobalSymbolAttacher {
 public:
  explicit GlobalSymbolAttacher(IRModule mod) : mod_(mod) {}

  IRModule Attach() {
    IRModule ret;
    for (auto& p : mod_->functions) {
      BaseFunc func = p.second;
      if (auto* prim_func = func.as<tir::PrimFuncNode>()) {
        func = WithAttr(GetRef<tir::PrimFunc>(prim_func), "global_symbol", p.first->name_hint);
      } else if (auto* relax_func = func.as<FunctionNode>()) {
        func = WithAttr(GetRef<Function>(relax_func), "global_symbol", p.first->name_hint);
      } else {
        LOG(FATAL) << "Unsupported function type: " << func->GetTypeKey();
        throw;
      }
      ret->Add(p.first, func);
    }
    return ret;
  }

 private:
  IRModule mod_;
};

namespace transform {

Pass AttachGlobalSymbol() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return GlobalSymbolAttacher(mod).Attach(); };
  return CreateModulePass(pass_func, 0, "AttachGlobalSymbol", {});
}

TVM_REGISTER_GLOBAL("relax.transform.AttachGlobalSymbol").set_body_typed(AttachGlobalSymbol);

}  // namespace transform

}  // namespace relax
}  // namespace tvm

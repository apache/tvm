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
#include <sstream>
#include <string>

#include "../../../../target/source/codegen_c_host.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace example_target_hooks {

using namespace tir;

class CodeGenExampleTargetHook : public codegen::CodeGenCHost {
 public:
  /*!
   * \brief Emit code that changes adds to multiplies for testing
   */
  void VisitExpr_(const SubNode* op, std::ostream& os) final {
    os << '(';
    PrintExpr(op->a, os);
    os << " * ";
    PrintExpr(op->b, os);
    os << ')';
  }
};

runtime::Module TIRToRuntime(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = false;
  CodeGenExampleTargetHook codegen;
  Array<String> function_names;
  codegen.Init(output_ssa, emit_asserts, target->str());
  for (auto kv : mod->functions) {
    auto prim_func = Downcast<PrimFunc>(kv.second);
    auto global_symbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    function_names.push_back(global_symbol.value());
    codegen.AddFunction(prim_func);
  }
  std::string code = codegen.Finish();
  return codegen::CSourceModuleCreate(code, "c", function_names);
}

}  // namespace example_target_hooks
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

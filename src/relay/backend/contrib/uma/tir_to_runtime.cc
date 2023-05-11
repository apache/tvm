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
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../../../runtime/file_utils.h"
#include "../../../../target/source/codegen_c.h"
#include "../../../../target/source/codegen_c_host.h"

namespace tvm {
using namespace tir;
namespace relay {
namespace contrib {
namespace uma {

class UMACodegen : public codegen::CodeGenCHost {
 public:
  explicit UMACodegen(String target_str) : target_str_(target_str) {}

  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl) {
    auto includes_pf =
        tvm::runtime::Registry::Get("relay.ext.uma.codegen_c_includes_" + target_str_);
    if (includes_pf) {
      String includes = (*includes_pf)();
      decl_stream << includes;
    }
    std::unordered_set<std::string> devices;
    devices.insert(target_str_);
    CodeGenCHost::Init(output_ssa, emit_asserts, emit_fwd_func_decl, target_str_, devices);
  }

  /*!
   * \brief Emit code that offloads a subgraph to the UMA target
   *
   * \return string of code that offloads a subgraph to the UMA target
   */
  void AddFunction(const PrimFunc& prim_func) { CodeGenC::AddFunction(prim_func); }

 private:
  String target_str_;
};

runtime::Module TIRToRuntime(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = false;
  UMACodegen codegen(target->kind->name);
  Array<String> function_names;
  codegen.Init(output_ssa, emit_asserts, emit_fwd_func_decl);
  for (auto kv : mod->functions) {
    auto prim_func = Downcast<PrimFunc>(kv.second);
    auto global_symbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    function_names.push_back(global_symbol.value());
    codegen.AddFunction(prim_func);
  }
  std::string code = codegen.Finish();
  return codegen::CSourceModuleCreate(code, "c", function_names);
}

}  // namespace uma
};  // namespace contrib
}  // namespace relay
}  // namespace tvm

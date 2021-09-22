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

namespace tvm {
namespace codegen {

using namespace tir;

class CodeGenCMSISNN : public CodeGenC {
 public:
  void Init(bool output_ssa) {
    decl_stream << "#include <stdio.h>\n";
    decl_stream << "#include <stdlib.h>\n";
    decl_stream << "#include <dlpack/dlpack.h>\n";
    decl_stream << "#include <tvm/runtime/crt/module.h>\n";
    decl_stream << "#include <arm_nnfunctions.h>\n";
    CodeGenC::Init(output_ssa);
  }

  /*!
   * \brief Emit code that offloads a subgraph to the Cortex-M
   *
   * \return string of code that offloads a subgraph to the Cortex-M
   */
  void AddFunction(const PrimFunc& prim_func) {
    PrintExternCPrefix(stream);
    CodeGenC::AddFunction(prim_func);
    PrintExternCPostfix(stream);
  }

 private:
  /*!  * \brief Creates a cplusplus guard prefix for extern "C" printing */
  void PrintExternCPrefix(std::ostringstream& ss) {
    PrintIndent();
    ss << "#ifdef __cplusplus\n";
    ss << "extern \"C\" {\n";
    ss << "#endif\n";
  }

  /*!  * \brief Creates a cplusplus guard postfix for extern "C" printing */
  void PrintExternCPostfix(std::ostringstream& ss) {
    PrintIndent();
    ss << "#ifdef __cplusplus\n";
    ss << "}\n";
    ss << "#endif\n";
  }
};

class CMSISNNModuleNode : public runtime::ModuleNode {
 public:
  CMSISNNModuleNode(const std::string& code, const std::string& fmt,
                    const Array<String>& func_names)
      : code_(code), fmt_(fmt), func_names_(func_names) {}

  std::string GetSource(const std::string& format) final { return code_; }

  const char* type_key() const { return "c"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->func_names_[0]; });
    } else if (name == "get_func_names") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->func_names_; });
    } else {
      return PackedFunc(nullptr);
    }
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    std::string meta_file = runtime::GetMetaFilePath(file_name);
    if (fmt == "c") {
      ICHECK_NE(code_.length(), 0);
      runtime::SaveBinaryToFile(file_name, code_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    }
  }

 protected:
  std::string code_;
  std::string fmt_;
  Array<String> func_names_;
};

static runtime::Module CMSISNNModuleNodeCreate(IRModule mod) {
  bool output_ssa = false;
  CodeGenCMSISNN cg;
  Array<String> function_names;
  cg.Init(output_ssa);
  ICHECK(mod->functions.size() == 1) << "Supports modules with single PrimFunc.";
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodegenCMSISNN: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.defined())
        << "CodeGenCHost: Expect PrimFunc to have the global_symbol attribute";
    function_names.push_back(global_symbol.value());
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();
  auto n = make_object<CMSISNNModuleNode>(code, "c", function_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.CMSISNNModuleNodeCreate").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CMSISNNModuleNodeCreate(args[0]);
});

}  // namespace codegen
}  // namespace tvm

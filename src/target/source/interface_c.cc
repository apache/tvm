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
 * \file interface_c.cc
 * \brief Generates a C interface header for a given modules inputs and outputs
 */

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <string>

#include "../../relay/backend/name_transforms.h"

namespace tvm {
namespace codegen {

using runtime::PackedFunc;
using namespace tvm::relay::backend;

class InterfaceCNode : public runtime::ModuleNode {
 public:
  InterfaceCNode(std::string module_name, Array<String> inputs, Array<String> outputs)
      : module_name_(module_name), inputs_(inputs), outputs_(outputs) {}
  const char* type_key() const { return "h"; }

  std::string GetSource(const std::string& format) final {
    std::stringstream code;

    EmitUpperHeaderGuard(code);
    EmitBrief(code, "Input tensor pointers");
    EmitStruct(code, "inputs", inputs_);
    EmitBrief(code, "Output tensor pointers");
    EmitStruct(code, "outputs", outputs_);
    EmitRunFunction(code);
    EmitLowerHeaderGuard(code);

    return code.str();
  }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    return PackedFunc(nullptr);
  }

 private:
  void EmitUpperHeaderGuard(std::stringstream& code_stream) {
    std::string header_guard_name = ToCConstantStyle(PrefixGeneratedName({module_name_, "H"}));
    code_stream << "#ifndef " << header_guard_name << "_\n"
                << "#define " << header_guard_name << "_\n"
                << "#include <stdint.h>\n\n"
                << "#ifdef __cplusplus\n"
                << "extern \"C\" {\n"
                << "#endif\n\n";
  }

  void EmitLowerHeaderGuard(std::stringstream& code_stream) {
    std::string header_guard_name = ToCConstantStyle(PrefixGeneratedName({module_name_, "H"}));
    code_stream << "\n#ifdef __cplusplus\n"
                << "}\n"
                << "#endif\n\n"
                << "#endif // " << header_guard_name << "_\n";
  }

  void EmitBrief(std::stringstream& code_stream, const std::string& description) {
    code_stream << "/*!\n"
                << " * \\brief " << description << " for TVM module \"" << module_name_ << "\" \n"
                << " */\n";
  }

  void EmitStruct(std::stringstream& code_stream, const std::string& suffix,
                  Array<String> properties) {
    std::string struct_name = ToCVariableStyle(PrefixGeneratedName({module_name_, suffix}));
    code_stream << "struct " << struct_name << " {\n";

    std::vector<std::string> sanitized_properties;
    for (const String& property : properties) {
      std::string sanitized_property = SanitizeName(property);
      ICHECK(std::find(sanitized_properties.begin(), sanitized_properties.end(),
                       sanitized_property) == sanitized_properties.end())
          << "Sanitized input tensor name clash" << sanitized_property;
      code_stream << "  void* " << sanitized_property << ";\n";
      sanitized_properties.push_back(sanitized_property);
    }
    code_stream << "};\n\n";
  }

  void EmitRunFunction(std::stringstream& code_stream) {
    std::string run_function = ToCVariableStyle(PrefixGeneratedName({module_name_, "run"}));
    std::string inputs_struct = ToCVariableStyle(PrefixGeneratedName({module_name_, "inputs"}));
    std::string outputs_struct = ToCVariableStyle(PrefixGeneratedName({module_name_, "outputs"}));

    code_stream << "/*!\n"
                << " * \\brief entrypoint function for TVM module \"" << module_name_ << "\"\n"
                << " * \\param inputs Input tensors for the module \n"
                << " * \\param outputs Output tensors for the module \n"
                << " */\n"
                << "int32_t " << run_function << "(\n"
                << "  struct " << inputs_struct << "* inputs,\n"
                << "  struct " << outputs_struct << "* outputs\n"
                << ");\n";
  }

  std::string module_name_;
  Array<String> inputs_;
  Array<String> outputs_;
};

runtime::Module InterfaceCCreate(std::string module_name, Array<String> inputs,
                                 Array<String> outputs) {
  auto n = make_object<InterfaceCNode>(module_name, inputs, outputs);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.InterfaceCCreate").set_body_typed(InterfaceCCreate);

}  // namespace codegen
}  // namespace tvm

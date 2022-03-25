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
#include <tvm/tir/usmp/utils.h>

#include <string>

#include "../../relay/backend/name_transforms.h"

namespace tvm {
namespace codegen {

using runtime::PackedFunc;
using namespace tvm::relay::backend;

class InterfaceCNode : public runtime::ModuleNode {
 public:
  InterfaceCNode(std::string module_name, Array<String> inputs, Array<String> outputs,
                 Array<tir::usmp::AllocatedPoolInfo> pools, Array<String> devices,
                 int workspace_size)
      : module_name_(module_name),
        inputs_(inputs),
        outputs_(outputs),
        devices_(devices),
        pools_(FilterExternalPools(pools)),
        workspace_size_(workspace_size) {}
  const char* type_key() const { return "h"; }

  std::string GetSource(const std::string& format) final {
    std::stringstream code;

    EmitUpperHeaderGuard(code);
    EmitBrief(code, "Input tensor pointers");
    EmitStruct(code, "inputs", inputs_);
    EmitBrief(code, "Output tensor pointers");
    EmitStruct(code, "outputs", outputs_);

    if (!devices_.empty()) {
      EmitBrief(code, "Device context pointers");
      EmitStruct(code, "devices", devices_);
    }
    if (!pools_.empty()) {
      EmitBrief(code, "Workspace pool pointers");
      Array<String> pool_names;
      for (const tir::usmp::AllocatedPoolInfo pool : pools_) {
        pool_names.push_back(pool->pool_info->pool_name);
      }
      EmitStruct(code, "workspace_pools", pool_names);
    }

    EmitRunFunction(code);
    // Emit workspace
    EmitIntegerValueMacro(code, "Workspace size", "WORKSPACE_SIZE", workspace_size_);
    // Emit memory pool sizes
    for (const tir::usmp::AllocatedPoolInfo pool : pools_) {
      String pool_name = pool->pool_info->pool_name;
      Integer pool_size = pool->allocated_size;
      EmitIntegerValueMacro(code, SanitizeName(pool_name) + " size",
                            SanitizeName(pool_name) + "_WORKSPACE_POOL_SIZE", pool_size->value);
    }
    EmitLowerHeaderGuard(code);

    return code.str();
  }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    return PackedFunc();
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

  void EmitIntegerValueMacro(std::stringstream& code_stream, const std::string& brief_description,
                             const std::string& macro_name, int macro_value) {
    EmitBrief(code_stream, brief_description);
    std::string macro_name_prefixed =
        ToCConstantStyle(PrefixGeneratedName({module_name_, macro_name}));
    code_stream << "#define " << macro_name_prefixed << " " << macro_value << "\n";
  }

  void EmitRunFunction(std::stringstream& code_stream) {
    std::string run_function = ToCVariableStyle(PrefixGeneratedName({module_name_, "run"}));
    std::string inputs_struct = ToCVariableStyle(PrefixGeneratedName({module_name_, "inputs"}));
    std::string outputs_struct = ToCVariableStyle(PrefixGeneratedName({module_name_, "outputs"}));
    std::string devices_struct = ToCVariableStyle(PrefixGeneratedName({module_name_, "devices"}));
    std::string pools_struct =
        ToCVariableStyle(PrefixGeneratedName({module_name_, "workspace_pools"}));

    code_stream << "/*!\n"
                << " * \\brief entrypoint function for TVM module \"" << module_name_ << "\"\n"
                << " * \\param inputs Input tensors for the module \n"
                << " * \\param outputs Output tensors for the module \n";

    if (!devices_.empty()) {
      code_stream << " * \\param devices Device context pointers for the module \n";
    }
    if (!pools_.empty()) {
      code_stream << " * \\param workspace_pools Workspace memory pool pointers for the module \n";
    }

    code_stream << " */\n"
                << "int32_t " << run_function << "(\n";

    std::stringstream call_args_ss;
    call_args_ss << "  struct " << inputs_struct << "* inputs,\n";
    call_args_ss << "  struct " << outputs_struct << "* outputs,\n";
    if (!devices_.empty()) {
      call_args_ss << "  struct " << devices_struct << "* devices,\n";
    }
    if (!pools_.empty()) {
      call_args_ss << "  struct " << pools_struct << "* workspace_pools,\n";
    }
    std::string call_args_str = call_args_ss.str();
    call_args_str.pop_back();
    call_args_str.pop_back();
    code_stream << call_args_str << "\n);\n";
  }

  Array<tir::usmp::AllocatedPoolInfo> FilterExternalPools(
      const Array<tir::usmp::AllocatedPoolInfo>& pools) {
    Array<tir::usmp::AllocatedPoolInfo> external_pools;
    for (tir::usmp::AllocatedPoolInfo pool : pools) {
      if (!pool->pool_info->is_internal) {
        external_pools.push_back(pool);
      }
    }
    return external_pools;
  }

  std::string module_name_;
  Array<String> inputs_;
  Array<String> outputs_;
  Array<String> devices_;
  Array<tir::usmp::AllocatedPoolInfo> pools_;
  int workspace_size_;
};

runtime::Module InterfaceCCreate(std::string module_name, Array<String> inputs,
                                 Array<String> outputs, Array<tir::usmp::AllocatedPoolInfo> pools,
                                 Array<String> devices, int workspace_size) {
  auto n =
      make_object<InterfaceCNode>(module_name, inputs, outputs, pools, devices, workspace_size);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.InterfaceCCreate").set_body_typed(InterfaceCCreate);

}  // namespace codegen
}  // namespace tvm

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
 * \file source_module.cc
 * \brief Source code module for the host to invoke the NPU
 */
#include <dmlc/filesystem.h>
#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../../../runtime/file_utils.h"
#include "utils.h"

namespace tvm {
namespace runtime {

using CompilationArtifact = relay::contrib::ethosu::CompilationArtifact;

// The runtime.Module that contains the host-side c code
// required for invoking the NPU with the command stream
class EthosUModuleNode : public ModuleNode {
 public:
  /*!
   * \brief The microNPU runtime module.
   *
   * \param compilation_artifacts
   *    This is an array of CompilationArtifacts that is produced via
   *    lowering each PrimFunc to command stream. Here, those artifacts
   *    will be used to create the c-source.
   */
  explicit EthosUModuleNode(Array<CompilationArtifact> compilation_artifacts)
      : compilation_artifacts_(compilation_artifacts) {
    c_source += "#include <stdio.h>\n";
    c_source += "#include <stdlib.h>\n";
    c_source += "#include <tvm/runtime/crt/module.h>\n";
    c_source += "#include <tvm_ethosu_runtime.h>\n\n";
    for (const CompilationArtifact& compilation_artifact : compilation_artifacts) {
      c_source += GenerateSource(compilation_artifact);
      c_source += "\n\n";
    }
  }

  /*!
   * \brief Save the module to file.
   *
   * \param file_name The file to be saved to.
   * \param format The format of the file.
   */
  void SaveToFile(const String& file_name, const String& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    ICHECK_EQ(fmt, "c") << "Can only save to format="
                        << "c";
    std::ofstream out(file_name);
    out << c_source;
    out.close();
  }

  String GetSource(const String& format) final { return c_source; }

  String GetFormat() override { return "c"; }

  Array<CompilationArtifact> GetArtifacts() { return compilation_artifacts_; }

  /*!
   * \brief Get a PackedFunc from the module.
   *
   * \param name The name of the function.
   * \param sptr_to_self The ObjectPtr that points to this module node.
   *
   * \return The function pointer when it is found, otherwise, PackedFunc(nullptr).
   */
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_func_names") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Array<String> func_names;
        for (const CompilationArtifact& ca : compilation_artifacts_) {
          func_names.push_back(ca->function_name);
        }
        *rv = func_names;
      });
    }
    return PackedFunc();
  }

  const char* type_key() const final { return "c"; }

  static Module Create(Array<CompilationArtifact> compilation_artifacts) {
    auto n = make_object<EthosUModuleNode>(compilation_artifacts);
    return Module(n);
  }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const override { return ModulePropertyMask::kDSOExportable; }

  bool ImplementsFunction(const String& name, bool query_imports) final {
    return std::find_if(compilation_artifacts_.begin(), compilation_artifacts_.end(),
                        [&name](const CompilationArtifact& artifact) {
                          return artifact->function_name == name;
                        }) != compilation_artifacts_.end();
  }

 private:
  std::string c_source;
  Array<CompilationArtifact> compilation_artifacts_;
  Map<Integer, String> pool_var_names_;
  int indent_{0};
  constexpr static int kMaxBaseAddresses_ = 6;

  /*!
   * \brief Convert the raw string of hex values into a hex string
   *
   * \param raw the raw string of hex values
   *
   * \return string formatted as a hex string
   */
  std::string GetHexString(const std::string& raw) {
    std::stringstream ss;
    for (size_t i = 0; i < raw.size() / 2; ++i) {
      ss << "\\x" << raw.substr(i * 2, 2);
    }
    return ss.str();
  }

  /*!
   * \brief Emit code that updates the base_addrs array with the base address of the given array
   *
   * \param index array index for base_addrs and base_addrs_size
   * \param name of the array containing relevant data
   *
   * \return string of code that updates the base_addrs array with the base address of the given
   * array
   */
  std::string SetBaseAddress(int index, std::string name, int size) {
    std::stringstream ss;
    ss << "  base_addrs[" << index << "] = (uintptr_t)(" << name << ");\n";
    ss << "  base_addrs_size[" << index << "] = " << size << ";\n";
    return ss.str();
  }

  /*!
   * \brief Enter a new scope.
   */
  void EnterScope() { indent_ += 2; }

  /*!
   * \brief Exit a scope.
   */
  void ExitScope() {
    ICHECK_GE(indent_, 2U) << "Wrong ident found.";
    indent_ -= 2;
  }

  /*! \brief Print indents using spaces. */
  void PrintIndents(std::stringstream& ss) {
    for (int i = 0; i < indent_; i++) {
      ss << ' ';
    }
  }

  /*!
   * \brief Creates a runtime function signature
   */
  void PrintRuntimeFunctionSignature(std::stringstream& ss,
                                     const relay::contrib::ethosu::CompilationArtifact& artifact,
                                     std::string func_name) {
    ss << "TVM_DLL int32_t " << func_name;
    ss << "(";
    std::unordered_map<int, relay::contrib::ethosu::BaseAddress> param_idx_to_base_address;
    for (const relay::contrib::ethosu::BaseAddress& base_address : artifact->base_addresses) {
      if (base_address->primfunc_param_idx.defined()) {
        param_idx_to_base_address[base_address->primfunc_param_idx.IntValue()] = base_address;
      }
    }
    for (unsigned int i = 0; i < param_idx_to_base_address.size(); i++) {
      relay::contrib::ethosu::BaseAddress base_address = param_idx_to_base_address[i];
      ss << "void* " << base_address->name << ",";
    }
    ss << "void* resource_handle) {\n";
  }

  /*!
   * \brief Creates a cplusplus guard prefix for extern "C" printing
   */
  void PrintExternCPrefix(std::stringstream& ss) {
    PrintIndents(ss);
    ss << "#ifdef __cplusplus\n";
    ss << "extern \"C\" {\n";
    ss << "#endif\n";
  }

  /*!
   * \brief Creates a cplusplus guard postfix for extern "C" printing
   */
  void PrintExternCPostfix(std::stringstream& ss) {
    PrintIndents(ss);
    ss << "#ifdef __cplusplus\n";
    ss << "}\n";
    ss << "#endif\n";
  }

  /*!
   * \brief Emit code that offloads a subgraph to the NPU
   *
   * \return string of code that offloads a subgraph to the NPU
   */
  std::string GenerateSource(relay::contrib::ethosu::CompilationArtifact compilation_artifact) {
    std::string func_no_dashes = compilation_artifact->function_name;
    std::replace(func_no_dashes.begin(), func_no_dashes.end(), '-', '_');
    std::stringstream ss;

    size_t weights_size = (compilation_artifact->encoded_constants.size() / 2);
    ss << "// Update linker script to place .rodata.tvm in memory that can be accessed by the "
          "NPU\n";
    if (weights_size > 0) {
      ss << "__attribute__((section(\".rodata.tvm\"), aligned(16))) static int8_t "
         << func_no_dashes << "_weights[" << weights_size << "] = \"";
      ss << GetHexString(compilation_artifact->encoded_constants);
      ss << "\";\n";
    } else {
      ss << "static int8_t* " << func_no_dashes << "_weights = NULL;\n";
    }
    ss << "__attribute__((section(\".rodata.tvm\"), aligned(16))) static int8_t " << func_no_dashes
       << "_cms_data_data[" << compilation_artifact->command_stream.size() / 2 << "] = \"";
    ss << GetHexString(compilation_artifact->command_stream);
    ss << "\";\n";
    ss << "\n";

    PrintExternCPrefix(ss);
    PrintRuntimeFunctionSignature(ss, compilation_artifact, func_no_dashes);
    ss << "  void* cms_data = (void*)(" << func_no_dashes << "_cms_data_data);\n";
    ss << "  const size_t cms_data_size = sizeof(" << func_no_dashes << "_cms_data_data);\n";
    ss << "  size_t base_addrs_size[" << kMaxBaseAddresses_ << "] = {0};\n";
    ss << "  uint64_t base_addrs[" << kMaxBaseAddresses_ << "] = {0};\n";
    ss << "\n";

    ss << SetBaseAddress(0, func_no_dashes + "_weights", weights_size);
    for (const relay::contrib::ethosu::BaseAddress& base_address :
         compilation_artifact->base_addresses) {
      if (base_address->is_runtime_allocation) {
        ss << "  int8_t* " << base_address->name
           << " = (int8_t*) TVMBackendAllocWorkspace(kDLCPU, 0, (uint64_t)" << base_address->size
           << ", 0, 16);\n";
      }
      ss << SetBaseAddress(base_address->region->value, base_address->name.c_str(),
                           base_address->size->value);
    }
    ss << "\n";

    ss << "  int32_t result = TVMEthosULaunch(resource_handle, cms_data, cms_data_size, "
          "base_addrs, base_addrs_size, "
       << kMaxBaseAddresses_ << ");\n";

    for (const relay::contrib::ethosu::BaseAddress& base_address :
         compilation_artifact->base_addresses) {
      if (base_address->is_runtime_allocation) {
        ss << "  TVMBackendFreeWorkspace(kDLCPU, 0, " << base_address->name << ");\n";
      }
    }
    ss << "  return result;\n";
    ss << "}\n";
    ss << "\n";
    PrintExternCPostfix(ss);
    ss << "\n";
    return ss.str();
  }
};

class EthosUModule : public Module {
 public:
  EthosUModule() {}
  explicit EthosUModule(ObjectPtr<Object> n) : Module(n) {}
  /*! \return internal container */
  inline EthosUModuleNode* operator->();
  /*! \return internal container */
  inline const EthosUModuleNode* operator->() const;
};

inline EthosUModuleNode* EthosUModule::operator->() {
  return static_cast<EthosUModuleNode*>(get_mutable());
}

TVM_REGISTER_GLOBAL("runtime.module.ethos-u.create")
    .set_body_typed([](Array<CompilationArtifact> compilation_artifacts) {
      return EthosUModuleNode::Create(compilation_artifacts);
    });

TVM_REGISTER_GLOBAL("runtime.module.ethos-u.get_artifacts").set_body_typed([](EthosUModule mod) {
  return mod->GetArtifacts();
});

}  // namespace runtime
}  // namespace tvm

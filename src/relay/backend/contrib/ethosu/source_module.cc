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

namespace tvm {
namespace runtime {

class EthosUModuleNode : public ModuleNode {
 public:
  /*!
   * \brief The ethos runtime module.
   *
   * \param cmms A array of external symbol 1, serialized command stream 1
   * external symbol 2, serialized command stream 2, ....
   * TODO : if and when FFI support Maps with non-objects OR compound arrays
   * switch to that.
   */
  explicit EthosUModuleNode(const String& func_name_, const String& cmms_hex_,
                            const String& weights_bias_hex_, const Integer& scratch_size_,
                            const Integer& input_size_, const Integer& output_size_) {
    func_names_.push_back(func_name_);
    cmms_hex = std::move(cmms_hex_);
    weights_bias_hex = std::move(weights_bias_hex_);
    scratch_size = scratch_size_->value;
    input_size = input_size_->value;
    output_size = output_size_->value;
    c_source = GenerateSource();
  }

  /*!
   * \brief Save the module to file.
   *
   * \param file_name The file to be saved to.
   * \param format The format of the file.
   */
  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    LOG(INFO) << "format=" << fmt << ";;\n";
    ICHECK_EQ(fmt, "c") << "Can only save to format="
                        << "c";
    std::ofstream out(file_name);
    out << c_source;
    out.close();
  }

  std::string GetSource(const std::string& format) final { return c_source; }

  std::string GetCS() { return cmms_hex; }

  /*!
   * \brief Get a PackedFunc from the module.
   *
   * \param name The name of the function.
   * \param sptr_to_self The ObjectPtr that points to this module node.
   *
   * \return The function pointer when it is found, otherwise, PackedFunc(nullptr).
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_func_names") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->func_names_; });
    }
    return PackedFunc();
  }

  const char* type_key() const override { return "c"; }

  static Module Create(String func_name, String cmms_hex, String weights_bias_hex,
                       Integer scratch_size, Integer input_size, Integer output_size) {
    auto n = make_object<EthosUModuleNode>(func_name, cmms_hex, weights_bias_hex, scratch_size,
                                           input_size, output_size);
    return Module(n);
  }

 private:
  String c_source;
  Array<String> func_names_;
  String cmms_hex;
  String weights_bias_hex;
  size_t scratch_size;
  size_t input_size;
  size_t output_size;
  int indent_{0};

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
  std::string SetBaseAddress(int index, std::string name) {
    std::stringstream ss;
    ss << "  base_addrs[" << index << "] = (uintptr_t)(" << name << ");\n";
    ss << "  base_addrs_size[" << index << "] = " << name << "_size;\n";
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
   * \brief Creates a runtime function header
   */
  void PrintRuntimeFunctionHeader(std::stringstream& ss, std::string func_name) {
    ss << "TVM_DLL int32_t ";
    ss << func_name << "(void* input, void* output) {\n";
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
  std::string GenerateSource() {
    std::string func_no_dashes = func_names_[0];
    std::replace(func_no_dashes.begin(), func_no_dashes.end(), '-', '_');
    std::stringstream ss;

    ss << "#include <stdio.h>\n";
    ss << "#include <stdlib.h>\n";
    ss << "#include <dlpack/dlpack.h>\n";
    ss << "#include <tvm/runtime/crt/module.h>\n";
    ss << "#include <ethosu_driver.h>\n";
    ss << "\n";
    size_t weights_size = (weights_bias_hex.size() / 2);
    ss << "static const size_t weights_size = " << std::to_string(weights_size) << ";\n";
    ss << "static const size_t scratch_size = " << std::to_string(scratch_size) << ";\n";
    ss << "// Update linker script to place ethosu_scratch in memory that can be accessed by the "
          "NPU\n";
    if (weights_size > 0) {
      ss << "__attribute__((section(\"ethosu_scratch\"), aligned(16))) static int8_t weights["
         << weights_size << "] = \"";
      ss << GetHexString(weights_bias_hex);
      ss << "\";\n";
    } else {
      ss << "static int8_t* weights = NULL;\n";
    }
    ss << "__attribute__((section(\"ethosu_scratch\"), aligned(16))) static int8_t cms_data_data["
       << cmms_hex.size() / 2 << "] = \"";
    ss << GetHexString(cmms_hex);
    ss << "\";\n";
    ss << "static const size_t cms_data_size = sizeof(cms_data_data);\n";
    ss << "\n";

    PrintExternCPrefix(ss);
    ss << "static int32_t " << func_no_dashes + "_(int8_t* in0, "
       << "size_t in0_size, int8_t* out0, size_t out0_size) {\n";
    ss << "  int num_tensors = 5;\n";
    ss << "  void* cms_data = (void*)(cms_data_data);\n";
    ss << "  int64_t device_type = kDLCPU;\n";
    ss << "  int64_t device_id = 0;\n";
    if (scratch_size > 0) {
      ss << "  int8_t* scratch = (int8_t*) TVMBackendAllocWorkspace(device_type, device_id, "
            "(uint64_t)scratch_size, 0, 16);\n";
    } else {
      ss << "  int8_t* scratch = NULL;\n";
    }
    ss << "  size_t base_addrs_size[num_tensors];\n";
    ss << "  uint64_t base_addrs[num_tensors];\n";
    ss << "\n";
    ss << SetBaseAddress(0, "weights");
    ss << SetBaseAddress(1, "scratch");
    ss << SetBaseAddress(2, "scratch");
    ss << SetBaseAddress(3, "in0");
    ss << SetBaseAddress(4, "out0");
    ss << "\n";
    ss << "  struct ethosu_driver *drv = ethosu_reserve_driver();\n";
    ss << "  int32_t result = ethosu_invoke(drv, cms_data, cms_data_size, base_addrs, "
          "base_addrs_size, "
          "num_tensors);\n";
    ss << "  ethosu_release_driver(drv);\n";
    if (scratch_size > 0) {
      ss << "  TVMBackendFreeWorkspace(device_type, device_id, scratch);\n";
    }
    ss << "  if (result != 0) {\n";
    ss << "    return -1;\n";
    ss << "  } else {\n";
    ss << "    return 0;\n";
    ss << "  }\n";
    ss << "}\n";
    ss << "\n";
    PrintExternCPostfix(ss);
    ss << "\n";
    PrintExternCPrefix(ss);
    ss << "// Wrapper function is provided to allow for easier debugging\n";
    ss << "inline static int32_t " + func_no_dashes + "_wrapper_(void* input, void* output) {\n";
    ss << "  size_t input_data_size = " << input_size << ";\n";
    ss << "  size_t output_data_size = " << output_size << ";\n";
    ss << "  return " + func_no_dashes +
              "_((int8_t*)input, input_data_size, (int8_t*)output, output_data_size);\n";
    ss << "}\n";
    PrintExternCPostfix(ss);
    ss << "\n";
    PrintExternCPrefix(ss);
    PrintRuntimeFunctionHeader(ss, func_names_[0]);
    EnterScope();
    PrintIndents(ss);
    ss << "return " << func_no_dashes << "_wrapper_(input, output);\n";
    ExitScope();
    ss << "}\n";
    PrintExternCPostfix(ss);

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

TVM_REGISTER_GLOBAL("runtime.module.ethosu.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = EthosUModuleNode::Create(args[0], args[1], args[2], args[3], args[4], args[5]);
});

TVM_REGISTER_GLOBAL("runtime.module.ethosu.getcs").set_body_typed([](EthosUModule mod) {
  return mod->GetCS();
});

}  // namespace runtime
}  // namespace tvm

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
 * \brief Source code module, only for viewing
 */
#include "source_module.h"

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <utility>

#include "../../runtime/file_utils.h"
#include "../../support/str_escape.h"
#include "../func_registry_generator.h"
#include "codegen_source_base.h"

namespace tvm {
namespace codegen {

using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

using runtime::FunctionInfo;
using runtime::GetFileFormat;
using runtime::GetMetaFilePath;
using runtime::SaveBinaryToFile;

// Simulator function
class SourceModuleNode : public runtime::ModuleNode {
 public:
  SourceModuleNode(std::string code, std::string fmt) : code_(code), fmt_(fmt) {}
  const char* type_key() const { return "source"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  std::string GetSource(const std::string& format) final { return code_; }

 protected:
  std::string code_;
  std::string fmt_;
};

runtime::Module SourceModuleCreate(std::string code, std::string fmt) {
  auto n = make_object<SourceModuleNode>(code, fmt);
  return runtime::Module(n);
}

// Simulator function
class CSourceModuleNode : public runtime::ModuleNode {
 public:
  CSourceModuleNode(const std::string& code, const std::string& fmt,
                    const Array<String>& func_names, const Array<String>& const_vars)
      : code_(code), fmt_(fmt), const_vars_(const_vars), func_names_(func_names) {}
  const char* type_key() const { return "c"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    // Currently c-source module is used as demonstration purposes with binary metadata module
    // that expects get_symbol interface. When c-source module is used as external module, it
    // will only contain one function. However, when its used as an internal module (e.g., target
    // "c") it can have many functions.
    if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->func_names_[0]; });
    } else if (name == "get_const_vars") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->const_vars_; });
    } else if (name == "get_func_names") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->func_names_; });
    } else {
      return PackedFunc(nullptr);
    }
  }

  std::string GetSource(const std::string& format) final { return code_; }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "c" || fmt == "cu") {
      ICHECK_NE(code_.length(), 0);
      SaveBinaryToFile(file_name, code_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    }
  }

 protected:
  std::string code_;
  std::string fmt_;
  Array<String> const_vars_;
  Array<String> func_names_;
};

runtime::Module CSourceModuleCreate(const String& code, const String& fmt,
                                    const Array<String>& func_names,
                                    const Array<String>& const_vars) {
  auto n = make_object<CSourceModuleNode>(code.operator std::string(), fmt.operator std::string(),
                                          func_names, const_vars);
  return runtime::Module(n);
}

class CSourceCrtMetadataModuleNode : public runtime::ModuleNode {
 public:
  CSourceCrtMetadataModuleNode(const Array<String>& func_names, const std::string& fmt,
                               Target target, runtime::Metadata metadata)
      : fmt_(fmt), func_names_(func_names), target_(target), metadata_(metadata) {
    CreateSource();
  }
  const char* type_key() const { return "c"; }

  std::string GetSource(const std::string& format) final { return code_.str(); }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    return PackedFunc(nullptr);
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "c") {
      auto code_str = code_.str();
      ICHECK_NE(code_str.length(), 0);
      SaveBinaryToFile(file_name, code_str);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    }
  }

 protected:
  std::stringstream code_;
  std::string fmt_;
  Array<String> func_names_;
  Target target_;
  runtime::Metadata metadata_;

  void CreateFuncRegistry() {
    code_ << "#include <tvm/runtime/crt/module.h>\n";
    for (const auto& fname : func_names_) {
      code_ << "#ifdef __cplusplus\n";
      code_ << "extern \"C\"\n";
      code_ << "#endif\n";
      code_ << "TVM_DLL int32_t " << fname.data();
      code_ << "(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* "
               "out_type_code);\n";
    }
    code_ << "static TVMBackendPackedCFunc _tvm_func_array[] = {\n";
    for (auto f : func_names_) {
      code_ << "    (TVMBackendPackedCFunc)" << f << ",\n";
    }
    code_ << "};\n";
    auto registry = target::GenerateFuncRegistryNames(func_names_);
    code_ << "static const TVMFuncRegistry _tvm_func_registry = {\n"
          << "    \"" << ::tvm::support::StrEscape(registry.data(), registry.size(), true) << "\","
          << "    _tvm_func_array,\n"
          << "};\n";
  }

  void GenerateCrtSystemLib() {
    code_ << "static const TVMModule _tvm_system_lib = {\n"
          << "    &_tvm_func_registry,\n"
          << "};\n"
          << "const TVMModule* TVMSystemLibEntryPoint(void) {\n"
          << "    return &_tvm_system_lib;\n"
          << "}\n";
  }

  void GenerateEntrypointForUnpackedAPI(const std::string& entrypoint_name,
                                        const std::string& run_func) {
    code_ << "TVM_DLL int32_t " << run_func << "(";
    int total_args = (metadata_->inputs.size() + metadata_->num_outputs);
    for (int i = 0; i < total_args; ++i) {
      code_ << "void* arg" << i << ",";
    }
    code_ << "void* resource_handle);\n";
    code_ << "int32_t " << entrypoint_name;
    code_ << "(void* args, void* type_code, int num_args, void* out_value, void* "
             "out_type_code, void* resource_handle) {\n";
    code_ << "return " << run_func << "(";
    for (int i = 0; i < total_args; ++i) {
      code_ << "((DLTensor*)(((TVMValue*)args)[" << i << "].v_handle))[0].data,";
    }
    code_ << "resource_handle);\n";
    code_ << "}\n";
  }

  void GenerateEntrypointForPackedAPI(const std::string& entrypoint_name,
                                      const std::string& run_func) {
    code_ << "TVM_DLL int32_t " << run_func;
    code_ << "(void* args, void* type_code, int num_args, void* out_value, void* "
             "out_type_code, void* resource_handle);\n";
    code_ << "int32_t " << entrypoint_name;
    code_ << "(void* args, void* type_code, int num_args, void* out_value, void* "
             "out_type_code, void* resource_handle) {\n";
    code_ << "return " << run_func;
    code_ << "(args, type_code, num_args, out_value, out_type_code, resource_handle);\n";
    code_ << "}\n";
  }

  void GenerateCInterfaceEntrypoint(const std::string& entrypoint_name, const std::string& run_func,
                                    const std::string& mod_name) {
    code_ << "#include <" << mod_name << ".h>\n";
    code_ << "TVM_DLL int32_t " << run_func << "(";
    unsigned int total_args = (metadata_->inputs.size() + metadata_->num_outputs);
    for (unsigned int i = 0; i < total_args; ++i) {
      code_ << "void* arg" << i;
      if (i + 1 != total_args) {
        code_ << ",";
      }
    }
    code_ << ");\n";
    code_ << "int32_t " << entrypoint_name << "(";
    code_ << "struct " << runtime::get_name_mangled(mod_name, "inputs") << "* inputs,"
          << "struct " << runtime::get_name_mangled(mod_name, "outputs") << "* outputs"
          << ") {";
    code_ << "return " << run_func << "(";
    for (const auto& input : metadata_->inputs) {
      code_ << "inputs->" << input << ",";
    }
    if (metadata_->num_outputs == 1) {
      code_ << "outputs->output";
    } else {
      for (int i = 0; i < metadata_->num_outputs; ++i) {
        code_ << "outputs->output" << i;
        if (i + 1 != metadata_->num_outputs) {
          code_ << ",";
        }
      }
    }
    code_ << ");\n";
    code_ << "}\n";
  }

  void GenerateAOTDescriptor() {
    const std::string run_func_suffix = ::tvm::runtime::symbol::tvm_run_func_suffix;
    const std::string tvm_entrypoint_suffix = ::tvm::runtime::symbol::tvm_entrypoint_suffix;
    const std::string run_func_mangled =
        runtime::get_name_mangled(metadata_->mod_name, run_func_suffix);
    const std::string entrypoint_mangled =
        runtime::get_name_mangled(metadata_->mod_name, tvm_entrypoint_suffix);
    const std::string network_mangled = runtime::get_name_mangled(metadata_->mod_name, "network");
    auto unpacked_api = target_->GetAttr<Bool>("unpacked-api").value_or(Bool(false));
    auto interface_api = target_->GetAttr<String>("interface-api").value_or(String("packed"));

    code_ << "#include \"tvm/runtime/c_runtime_api.h\"\n";
    code_ << "#ifdef __cplusplus\n";
    code_ << "extern \"C\" {\n";
    code_ << "#endif\n";

    if (unpacked_api) {
      if (interface_api == "c") {
        GenerateCInterfaceEntrypoint(entrypoint_mangled, run_func_mangled, metadata_->mod_name);
      } else {
        GenerateEntrypointForUnpackedAPI(entrypoint_mangled, run_func_mangled);
      }
    } else {
      ICHECK_EQ(interface_api, "packed") << "Packed interface required for packed operators";
      GenerateEntrypointForPackedAPI(entrypoint_mangled, run_func_mangled);
    }

    code_ << "#ifdef __cplusplus\n";
    code_ << "}\n";
    code_ << "#endif\n";
  }

  void CreateSource() {
    if (target_->GetAttr<Bool>("system-lib").value_or(Bool(false)) && !func_names_.empty()) {
      CreateFuncRegistry();
      GenerateCrtSystemLib();
    }
    if (metadata_.defined() && metadata_->executor == runtime::kTvmExecutorAot) {
      GenerateAOTDescriptor();
    }
    code_ << ";";
  }
};

runtime::Module CreateCSourceCrtMetadataModule(const Array<runtime::Module>& modules, Target target,
                                               runtime::Metadata metadata) {
  Array<String> func_names;
  for (runtime::Module mod : modules) {
    auto pf_funcs = mod.GetFunction("get_func_names");
    if (pf_funcs != nullptr) {
      Array<String> func_names_ = pf_funcs();
      for (const auto& fname : func_names_) {
        func_names.push_back(fname);
      }
    }
  }
  auto n = make_object<CSourceCrtMetadataModuleNode>(func_names, "cc", target, metadata);
  auto csrc_metadata_module = runtime::Module(n);
  for (const auto& mod : modules) {
    csrc_metadata_module.Import(mod);
  }
  return std::move(csrc_metadata_module);
}

// supports limited save without cross compile
class DeviceSourceModuleNode final : public runtime::ModuleNode {
 public:
  DeviceSourceModuleNode(std::string data, std::string fmt,
                         std::unordered_map<std::string, FunctionInfo> fmap, std::string type_key,
                         std::function<std::string(const std::string&)> fget_source)
      : data_(data), fmt_(fmt), fmap_(fmap), type_key_(type_key), fget_source_(fget_source) {}

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  std::string GetSource(const std::string& format) final {
    if (fget_source_ != nullptr) {
      return fget_source_(format);
    } else {
      return data_;
    }
  }

  const char* type_key() const { return type_key_.c_str(); }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    std::string meta_file = GetMetaFilePath(file_name);
    SaveMetaDataToFile(meta_file, fmap_);
    SaveBinaryToFile(file_name, data_);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

 private:
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
  std::string type_key_;
  std::function<std::string(const std::string&)> fget_source_;
};

runtime::Module DeviceSourceModuleCreate(
    std::string data, std::string fmt, std::unordered_map<std::string, FunctionInfo> fmap,
    std::string type_key, std::function<std::string(const std::string&)> fget_source) {
  auto n = make_object<DeviceSourceModuleNode>(data, fmt, fmap, type_key, fget_source);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.SourceModuleCreate").set_body_typed(SourceModuleCreate);

TVM_REGISTER_GLOBAL("runtime.CSourceModuleCreate")
    .set_body_typed([](String code, String fmt, Array<String> func_names,
                       Array<String> const_vars) {
      return CSourceModuleCreate(code, fmt, func_names, const_vars);
    });

TVM_REGISTER_GLOBAL("runtime.CreateCSourceCrtMetadataModule")
    .set_body_typed([](const Array<runtime::Module>& modules, Target target) {
      // Note that we don't need metadata when we compile a single operator
      return CreateCSourceCrtMetadataModule(modules, target, runtime::Metadata());
    });

}  // namespace codegen
}  // namespace tvm

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

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../relay/backend/name_transforms.h"
#include "../../runtime/file_utils.h"
#include "../../support/str_escape.h"
#include "../func_registry_generator.h"
#include "../metadata.h"
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

  std::string GetFormat() { return fmt_; }

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

  std::string GetFormat() { return fmt_; }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "c" || fmt == "cc" || fmt == "cpp" || fmt == "cu") {
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

/*!
 * \brief A concrete class to get access to base methods of CodegenSourceBase.
 *
 * This class exist to get access to methods of CodegenSourceBase without duplicating
 * them. Therefore, keeping alignment with how codegen and source_module here generates
 * code.
 */
class ConcreteCodegenSourceBase : public CodeGenSourceBase {
  /*!
   * \brief Do nothing as this class exist to get access to methods of CodeGenSourceBase
   */
  void PrintSSAAssign(const std::string& target, const std::string& src, DataType t) final {
    return;
  }
};

class CSourceCrtMetadataModuleNode : public runtime::ModuleNode {
 public:
  CSourceCrtMetadataModuleNode(const Array<String>& func_names, const std::string& fmt,
                               Target target, relay::Runtime runtime,
                               relay::backend::ExecutorCodegenMetadata metadata)
      : fmt_(fmt),
        func_names_(func_names),
        target_(target),
        runtime_(runtime),
        metadata_(metadata) {
    CreateSource();
  }
  const char* type_key() const { return "c"; }

  std::string GetSource(const std::string& format) final { return code_.str(); }

  std::string GetFormat() { return fmt_; }
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    return PackedFunc(nullptr);
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "c" || fmt == "cc" || fmt == "cpp") {
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
  relay::Runtime runtime_;
  relay::backend::ExecutorCodegenMetadata metadata_;
  ConcreteCodegenSourceBase codegen_c_base_;

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

  String GenerateDLTensorStructWrapper(String reference_arg) {
    code_ << "DLTensor " << reference_arg << "_dltensor = {\n";
    code_ << ".data = &" << reference_arg << "\n";
    code_ << "};\n";
    code_ << "TVMValue " << reference_arg << "_tvm_value = {\n";
    code_ << ".v_handle = &" << reference_arg << "_dltensor\n";
    code_ << "};\n";
    return reference_arg + "_tvm_value";
  }

  void GenerateInternalWorkspaceBuffers() {
    if (metadata_->pool_inputs.defined()) {
      for (const auto& kv : metadata_->pool_inputs.value()) {
        tir::usmp::AllocatedPoolInfo allocated_pool_info = kv.second;
        if (allocated_pool_info->pool_info->is_internal) {
          code_ << "__attribute__((section(\".data.tvm\"), ";
          code_ << "aligned(" << 16 << ")))\n";
          code_ << "static uint8_t " << allocated_pool_info->pool_info->pool_name << "["
                << allocated_pool_info->allocated_size->value << "];\n";
        }
      }
    }
  }

  bool IsInternalWorkspaceBuffer(const tir::Var& pool_var) {
    if (metadata_->pool_inputs.defined()) {
      Map<tir::Var, tir::usmp::AllocatedPoolInfo> allocated_pool_infos =
          metadata_->pool_inputs.value();
      if (allocated_pool_infos.find(pool_var) != allocated_pool_infos.end()) {
        tir::usmp::AllocatedPoolInfo allocate_pool_info = allocated_pool_infos[pool_var];
        if (allocate_pool_info->pool_info->is_internal) {
          return true;
        }
      }
    }
    return false;
  }

  void GenerateEntrypointForUnpackedAPI(const std::string& entrypoint_name,
                                        const std::string& run_func) {
    code_ << "TVM_DLL int32_t " << run_func << "(";

    {
      std::stringstream call_args_ss;
      for (const tir::Var& input_var : metadata_->inputs) {
        if (input_var->type_annotation.defined()) {
          codegen_c_base_.PrintType(input_var->type_annotation, call_args_ss);
        } else {
          codegen_c_base_.PrintType(input_var.dtype(), call_args_ss);
        }
        call_args_ss << " " << input_var->name_hint << ",";
      }
      for (unsigned int i = 0; i < metadata_->outputs.size(); ++i) {
        call_args_ss << "void* output" << i << ",";
      }
      for (const tir::Var& pool_var : metadata_->pools) {
        if (pool_var->type_annotation.defined()) {
          codegen_c_base_.PrintType(pool_var->type_annotation, call_args_ss);
        } else {
          codegen_c_base_.PrintType(pool_var.dtype(), call_args_ss);
        }
        call_args_ss << " " << pool_var->name_hint << ",";
      }
      std::string call_args_str = call_args_ss.str();
      call_args_str.pop_back();
      code_ << call_args_str;
    }

    code_ << ");\n";
    code_ << "int32_t " << entrypoint_name;
    code_ << "(void* args, void* type_code, int num_args, void* out_value, void* "
             "out_type_code, void* resource_handle) {\n";
    code_ << "return " << run_func << "(";

    {
      std::stringstream call_args_ss;
      for (unsigned int i = 0; i < metadata_->inputs.size(); ++i) {
        call_args_ss << "((DLTensor*)(((TVMValue*)args)[" << i << "].v_handle))[0].data,";
      }
      for (unsigned int i = 0; i < metadata_->outputs.size(); ++i) {
        int j = metadata_->inputs.size() + i;
        call_args_ss << "((DLTensor*)(((TVMValue*)args)[" << j << "].v_handle))[0].data,";
      }
      for (const tir::Var& pool_var : metadata_->pools) {
        if (IsInternalWorkspaceBuffer(pool_var)) {
          call_args_ss << "&" << metadata_->pool_inputs.value()[pool_var]->pool_info->pool_name
                       << ",";
        }
      }
      std::string call_args_str = call_args_ss.str();
      call_args_str.pop_back();
      code_ << call_args_str;
      code_ << ");\n";
      code_ << "}\n";
    }
  }

  std::unordered_map<int, ObjectRef> GenerateRunFuncToEntryPointArgMap() {
    std::unordered_map<int, ObjectRef> run_func_to_entry_point_args;
    int entrypoint_arg_count = 0;
    int run_func_arg_count = 0;

    for (unsigned int i = 0; i < metadata_->inputs.size(); i++) {
      run_func_to_entry_point_args[run_func_arg_count] = Integer(entrypoint_arg_count);
      entrypoint_arg_count++;
      run_func_arg_count++;
    }
    for (unsigned int i = 0; i < metadata_->outputs.size(); i++) {
      run_func_to_entry_point_args[run_func_arg_count] = Integer(entrypoint_arg_count);
      entrypoint_arg_count++;
      run_func_arg_count++;
    }
    for (const tir::Var& pool_var : metadata_->pools) {
      if (IsInternalWorkspaceBuffer(pool_var)) {
        tir::usmp::AllocatedPoolInfo allocated_pool_info = metadata_->pool_inputs.value()[pool_var];
        run_func_to_entry_point_args[run_func_arg_count] =
            allocated_pool_info->pool_info->pool_name;
        run_func_arg_count++;
      }
    }
    return run_func_to_entry_point_args;
  }

  void GenerateEntrypointForPackedAPI(const std::string& entrypoint_name,
                                      const std::string& run_func) {
    code_ << "TVM_DLL int32_t " << run_func;
    code_ << "(void* args, void* type_code, int num_args, void* out_value, void* "
             "out_type_code, void* resource_handle);\n\n";

    code_ << "int32_t " << entrypoint_name;
    code_ << "(void* args, void* type_code, int num_args, void* out_value, void* "
             "out_type_code, void* resource_handle) {\n";

    // We are creating a copy of the set of pointers
    size_t number_of_io_tensors =
        metadata_->inputs.size() + metadata_->outputs.size() + metadata_->pools.size();
    code_ << "TVMValue tensors[" << number_of_io_tensors << "];\n";

    std::unordered_map<int, ObjectRef> run_func_to_entry_point_args =
        GenerateRunFuncToEntryPointArgMap();
    for (unsigned int i = 0; i < number_of_io_tensors; i++) {
      if (run_func_to_entry_point_args.find(i) != run_func_to_entry_point_args.end()) {
        if (run_func_to_entry_point_args[i]->IsInstance<StringObj>()) {
          String pool_name = Downcast<String>(run_func_to_entry_point_args[i]);
          String pool_name_tvmv = GenerateDLTensorStructWrapper(pool_name);
          code_ << "tensors[" << i << "] = " << pool_name_tvmv << ";\n";
        } else {
          code_ << "tensors[" << i << "] = ((TVMValue*)args)["
                << run_func_to_entry_point_args[Integer(i)] << "];\n";
        }
      }
    }

    code_ << "return " << run_func;
    code_ << "((void*)tensors, type_code, num_args, out_value, out_type_code, resource_handle);\n";
    code_ << "}\n";
  }

  static int isNotAlnum(char c) { return !std::isalnum(c); }

  void GenerateCInterfaceEntrypoint(const std::string& entrypoint_name, const std::string& run_func,
                                    const std::string& mod_name) {
    code_ << "#include <" << mod_name << ".h>\n";
    code_ << "TVM_DLL int32_t " << run_func << "(";
    {
      std::stringstream call_args_ss;
      for (const tir::Var& input_var : metadata_->inputs) {
        if (input_var->type_annotation.defined()) {
          codegen_c_base_.PrintType(input_var->type_annotation, call_args_ss);
        } else {
          codegen_c_base_.PrintType(input_var.dtype(), call_args_ss);
        }
        call_args_ss << " " << relay::backend::SanitizeName(input_var->name_hint) << ",";
      }
      for (unsigned int i = 0; i < metadata_->outputs.size(); ++i) {
        call_args_ss << "void* output" << i << ",";
      }
      for (const tir::Var& pool_var : metadata_->pools) {
        if (pool_var->type_annotation.defined()) {
          codegen_c_base_.PrintType(pool_var->type_annotation, call_args_ss);
        } else {
          codegen_c_base_.PrintType(pool_var.dtype(), call_args_ss);
        }
        call_args_ss << " " << pool_var->name_hint << ",";
      }
      for (const String& device : metadata_->devices) {
        call_args_ss << "void* " << device << ",";
      }
      std::string call_args_str = call_args_ss.str();
      call_args_str.pop_back();
      code_ << call_args_str;
    }

    code_ << ");\n";
    code_ << "int32_t " << entrypoint_name << "(";
    {
      std::stringstream call_args_ss;
      call_args_ss << "struct " << runtime::get_name_mangled(mod_name, "inputs") << "* inputs,";
      call_args_ss << "struct " << runtime::get_name_mangled(mod_name, "outputs") << "* outputs,";
      if (!metadata_->pools.empty()) {
        bool is_external_pools_present = false;
        for (tir::Var pool_var : metadata_->pools) {
          if (!IsInternalWorkspaceBuffer(pool_var)) {
            is_external_pools_present = true;
            break;
          }
        }
        if (is_external_pools_present) {
          call_args_ss << "struct " << runtime::get_name_mangled(mod_name, "workspace_pools")
                       << "* workspace_pools,";
        }
      }
      if (!metadata_->devices.empty()) {
        call_args_ss << "struct " << runtime::get_name_mangled(mod_name, "devices") << "* devices,";
      }
      std::string call_args_str = call_args_ss.str();
      call_args_str.pop_back();
      code_ << call_args_str;
    }

    code_ << ") {"
          << "return " << run_func << "(";

    {
      std::stringstream call_args_ss;
      for (const auto& input : metadata_->inputs) {
        call_args_ss << "inputs->" << relay::backend::SanitizeName(input->name_hint) << ",";
      }
      for (const auto& output : metadata_->outputs) {
        call_args_ss << "outputs->" << relay::backend::SanitizeName(output);
        call_args_ss << ",";
      }

      for (const tir::Var& pool_var : metadata_->pools) {
        String pool_name = metadata_->pool_inputs.value()[pool_var]->pool_info->pool_name;
        if (IsInternalWorkspaceBuffer(pool_var)) {
          call_args_ss << "&" << pool_name << ",";
        } else {
          call_args_ss << "workspace_pools->" << relay::backend::SanitizeName(pool_name) << ",";
        }
      }
      for (const String& device : metadata_->devices) {
        call_args_ss << "devices->" << device << ",";
      }
      std::string call_args_str = call_args_ss.str();
      call_args_str.pop_back();
      code_ << call_args_str;
    }
    code_ << ");\n";
    code_ << "}\n";
  }

  void GenerateAOTDescriptor() {
    const std::string run_func_suffix = ::tvm::runtime::symbol::tvm_module_main;
    const std::string tvm_entrypoint_suffix = ::tvm::runtime::symbol::tvm_entrypoint_suffix;
    const std::string run_func_mangled =
        runtime::get_name_mangled(metadata_->mod_name, run_func_suffix);
    const std::string entrypoint_mangled =
        runtime::get_name_mangled(metadata_->mod_name, tvm_entrypoint_suffix);
    const std::string network_mangled = runtime::get_name_mangled(metadata_->mod_name, "network");

    code_ << "#include \"tvm/runtime/c_runtime_api.h\"\n";
    code_ << "#ifdef __cplusplus\n";
    code_ << "extern \"C\" {\n";
    code_ << "#endif\n";

    GenerateInternalWorkspaceBuffers();

    if (metadata_->unpacked_api) {
      if (metadata_->interface_api == "c") {
        GenerateCInterfaceEntrypoint(entrypoint_mangled, run_func_mangled, metadata_->mod_name);
      } else {
        GenerateEntrypointForUnpackedAPI(entrypoint_mangled, run_func_mangled);
      }
    } else {
      ICHECK_EQ(metadata_->interface_api, "packed")
          << "Packed interface required for packed operators";
      GenerateEntrypointForPackedAPI(entrypoint_mangled, run_func_mangled);
    }

    code_ << "#ifdef __cplusplus\n";
    code_ << "}\n";
    code_ << "#endif\n";
  }

  void CreateSource() {
    if (runtime_->GetAttr<Bool>("system-lib").value_or(Bool(false)) && !func_names_.empty()) {
      CreateFuncRegistry();
      GenerateCrtSystemLib();
    }
    if (metadata_.defined() && metadata_->executor == runtime::kTvmExecutorAot) {
      GenerateAOTDescriptor();
    }
    code_ << ";";
  }
};

static std::string address_from_parts(const std::vector<std::string>& parts) {
  std::stringstream ss;
  for (unsigned int i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      ss << "_";
    }
    ss << parts[i];
  }
  return ss.str();
}

class MetadataQueuer : public AttrVisitor {
 public:
  using QueueItem = std::tuple<std::string, runtime::metadata::MetadataBase>;
  explicit MetadataQueuer(std::vector<QueueItem>* queue) : queue_{queue} {}

  void Visit(const char* key, double* value) final {}
  void Visit(const char* key, int64_t* value) final {}
  void Visit(const char* key, uint64_t* value) final {}
  void Visit(const char* key, int* value) final {}
  void Visit(const char* key, bool* value) final {}
  void Visit(const char* key, std::string* value) final {}
  void Visit(const char* key, DataType* value) final {}
  void Visit(const char* key, runtime::NDArray* value) final {}
  void Visit(const char* key, void** value) final {}

  void Visit(const char* key, ObjectRef* value) final {
    address_parts_.push_back(key);
    if (value->as<runtime::metadata::MetadataBaseNode>() != nullptr) {
      auto metadata = Downcast<runtime::metadata::MetadataBase>(*value);
      const runtime::metadata::MetadataArrayNode* arr =
          value->as<runtime::metadata::MetadataArrayNode>();
      if (arr != nullptr) {
        for (unsigned int i = 0; i < arr->array.size(); i++) {
          ObjectRef o = arr->array[i];
          if (o.as<runtime::metadata::MetadataBaseNode>() != nullptr) {
            std::stringstream ss;
            ss << i;
            address_parts_.push_back(ss.str());
            runtime::metadata::MetadataBase metadata = Downcast<runtime::metadata::MetadataBase>(o);
            ReflectionVTable::Global()->VisitAttrs(metadata.operator->(), this);
            address_parts_.pop_back();
          }
        }
      } else {
        ReflectionVTable::Global()->VisitAttrs(metadata.operator->(), this);
      }

      queue_->push_back(std::make_tuple(address_from_parts(address_parts_),
                                        Downcast<runtime::metadata::MetadataBase>(*value)));
    }
    address_parts_.pop_back();
  }

 private:
  std::vector<QueueItem>* queue_;
  std::vector<std::string> address_parts_;
};

class MetadataSerializer : public AttrVisitor {
 public:
  static constexpr const char* kGlobalSymbol = "kTvmgenMetadata";
  using MetadataTypeIndex = ::tvm::runtime::metadata::MetadataTypeIndex;

  MetadataSerializer() : is_first_item_{true} {}

  void WriteComma() {
    if (is_first_item_) {
      is_first_item_ = false;
    } else {
      code_ << ", " << std::endl;
    }
  }

  void WriteKey(const char* key) {
    if (key != nullptr) {
      code_ << " /* " << key << "*/";
    }
  }

  void Visit(const char* key, double* value) final {
    WriteComma();
    code_.setf(std::ios::hex | std::ios::showbase | std::ios::fixed | std::ios::scientific,
               std::ios::basefield | std::ios::showbase | std::ios::floatfield);
    code_ << *value;
    WriteKey(key);
  }

  void Visit(const char* key, int64_t* value) final {
    WriteComma();
    code_ << *value << "L";
    WriteKey(key);
  }

  void Visit(const char* key, uint64_t* value) final {
    WriteComma();
    code_ << *value << "UL";
    WriteKey(key);
  }
  void Visit(const char* key, int* value) final {
    WriteComma();
    code_ << *value;
    WriteKey(key);
  }
  void Visit(const char* key, bool* value) final {
    WriteComma();
    code_ << *value;
    WriteKey(key);
  }
  void Visit(const char* key, std::string* value) final {
    WriteComma();
    code_ << "\"" << *value << "\"";
    WriteKey(key);
  }
  void Visit(const char* key, void** value) final {
    WriteComma();
    code_ << *value;
    WriteKey(key);
  }
  void Visit(const char* key, DataType* value) final {
    WriteComma();
    code_ << "{" << value->code() << ", " << value->bits() << ", " << value->lanes() << "}";
    WriteKey(key);
  }

  void Visit(const char* key, runtime::NDArray* value) final {
    // TODO(areusch): probably we could consolidate --link-params here, tho...
    ICHECK(false) << "do not support serializing NDArray as metadata";
  }

  void VisitArray(const runtime::metadata::MetadataArrayNode* array) {
    auto old_is_first_item = is_first_item_;
    is_first_item_ = true;
    for (unsigned int i = 0; i < array->array.size(); ++i) {
      ObjectRef o = array->array[i];
      if (o->IsInstance<IntImmNode>()) {
        int64_t i = Downcast<Integer>(o);
        Visit(nullptr, &i);
        continue;
      }

      if (o->IsInstance<StringObj>()) {
        std::string s = Downcast<String>(o);
        Visit(nullptr, &s);
        continue;
      }

      runtime::metadata::MetadataBase metadata = Downcast<runtime::metadata::MetadataBase>(o);
      std::stringstream i_str;
      i_str << i;
      address_.push_back(i_str.str());
      Visit(nullptr, &metadata);
      address_.pop_back();
    }
    is_first_item_ = old_is_first_item;
  }

  void Visit(const char* key, ObjectRef* value) final {
    const runtime::metadata::MetadataArrayNode* arr =
        value->as<runtime::metadata::MetadataArrayNode>();
    if (arr != nullptr) {
      WriteComma();
      if (key != nullptr) {
        address_.push_back(key);
      }
      code_ << address_from_parts(address_);
      if (key != nullptr) {
        address_.pop_back();
      }
      return;
    }

    runtime::metadata::MetadataBase metadata = Downcast<runtime::metadata::MetadataBase>(*value);
    if (key != nullptr) {  // NOTE: outermost call passes nullptr key
      address_.push_back(key);
    }
    ReflectionVTable::Global()->VisitAttrs(metadata.operator->(), this);
    if (key != nullptr) {  // NOTE: outermost call passes nullptr key
      address_.pop_back();
    }
  }

  void CodegenMetadata(::tvm::runtime::metadata::Metadata metadata) {
    decl_ << "#include <inttypes.h>" << std::endl
          << "#include <tvm/runtime/metadata.h>" << std::endl
          << "#include <tvm/runtime/c_runtime_api.h>" << std::endl;
    std::vector<MetadataQueuer::QueueItem> queue;
    MetadataQueuer queuer{&queue};
    queuer.Visit(kGlobalSymbol, &metadata);

    for (MetadataQueuer::QueueItem item : queue) {
      auto struct_name = std::get<0>(item);
      auto obj = std::get<1>(item);
      auto arr = obj.as<runtime::metadata::MetadataArrayNode>();
      is_first_item_ = true;
      address_.push_back(struct_name);
      if (arr != nullptr) {
        const char* const_part = "const ";
        if (arr->type_index == MetadataTypeIndex::kString) {
          const_part = "";
        }
        code_ << const_part;
        switch (arr->type_index) {
          case MetadataTypeIndex::kUint64:
            code_ << "uint64_t";
            break;
          case MetadataTypeIndex::kInt64:
            code_ << "int64_t";
            break;
          case MetadataTypeIndex::kBool:
            code_ << "bool";
            break;
          case MetadataTypeIndex::kString:
            code_ << "const char*";
            break;
          case MetadataTypeIndex::kHandle:
            code_ << "void*";
            break;
          case MetadataTypeIndex::kMetadata:
            code_ << "struct " << arr->struct_name;
            break;
          default:
            CHECK(false) << "Unknown type_index in array: " << arr->type_index
                         << " (struct_name=" << arr->struct_name << ")";
            break;
        }
        code_ << " " << struct_name << "[" << arr->array.size() << "] = {" << std::endl;
        VisitArray(arr);
      } else {
        code_ << "const struct TVMMetadata " << struct_name << " = {" << std::endl;
        Visit(nullptr, &obj);
      }
      address_.pop_back();
      code_ << "};" << std::endl;
    }
  }

  std::string GetOutput() { return decl_.str() + code_.str(); }

 private:
  std::vector<std::string> address_;
  std::stringstream decl_;
  std::stringstream code_;
  bool is_first_item_;
  std::unordered_set<std::string> generated_struct_decls_;
  std::vector<bool> is_defining_struct_;
};

runtime::Module CreateCSourceCrtMetadataModule(const Array<runtime::Module>& modules, Target target,
                                               relay::Runtime runtime,
                                               relay::backend::ExecutorCodegenMetadata metadata) {
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
  auto n = make_object<CSourceCrtMetadataModuleNode>(func_names, "c", target, runtime, metadata);
  auto csrc_metadata_module = runtime::Module(n);
  for (const auto& mod : modules) {
    csrc_metadata_module.Import(mod);
  }
  return std::move(csrc_metadata_module);
}

runtime::Module CreateCSourceCppMetadataModule(runtime::metadata::Metadata metadata) {
  MetadataSerializer serializer;
  serializer.CodegenMetadata(metadata);
  std::stringstream lookup_func;
  lookup_func << "#ifdef __cplusplus\n"
              << "extern \"C\"\n"
              << "#endif\n";

  lookup_func << "TVM_DLL int32_t " << ::tvm::runtime::symbol::tvm_get_c_metadata
              << "(TVMValue* arg_values, int* arg_tcodes, int "
                 "num_args, TVMValue* ret_values, int* ret_tcodes, void* resource_handle) {"
              << std::endl;
  lookup_func << "    ret_values[0].v_handle = (void*) &" << MetadataSerializer::kGlobalSymbol
              << ";" << std::endl;
  lookup_func << "    ret_tcodes[0] = kTVMOpaqueHandle;" << std::endl;
  lookup_func << "    return 0;" << std::endl;
  lookup_func << "};" << std::endl;

  auto mod = MetadataModuleCreate(metadata);
  std::vector<String> func_names{::tvm::runtime::symbol::tvm_get_c_metadata};
  auto c = CSourceModuleCreate(serializer.GetOutput() + lookup_func.str(), "c", func_names,
                               Array<String>());
  mod->Import(c);
  return mod;
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
    .set_body_typed([](const Array<runtime::Module>& modules, Target target,
                       relay::Runtime runtime) {
      // Note that we don't need metadata when we compile a single operator
      return CreateCSourceCrtMetadataModule(modules, target, runtime,
                                            relay::backend::ExecutorCodegenMetadata());
    });

}  // namespace codegen
}  // namespace tvm

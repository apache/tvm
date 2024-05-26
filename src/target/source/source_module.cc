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

#include <dmlc/memory_io.h>
#include <tvm/runtime/metadata.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/name_transforms.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../relay/backend/name_transforms.h"
#include "../../runtime/file_utils.h"
#include "../../support/str_escape.h"
#include "../func_registry_generator.h"
#include "../metadata.h"
#include "../metadata_utils.h"
#include "codegen_params.h"
#include "codegen_source_base.h"
#include "tvm/relay/executor.h"

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
  const char* type_key() const final { return "source"; }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  String GetSource(const String& format) final { return code_; }

  String GetFormat() override { return fmt_; }

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
  const char* type_key() const final { return "c"; }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
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

  String GetSource(const String& format) final { return code_; }

  String GetFormat() override { return fmt_; }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(code_);
    stream->Write(fmt_);

    std::vector<std::string> func_names;
    for (const auto func_name : func_names_) func_names.push_back(func_name);
    std::vector<std::string> const_vars;
    for (auto const_var : const_vars_) const_vars.push_back(const_var);
    stream->Write(func_names);
    stream->Write(const_vars);
  }

  static runtime::Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);

    std::string code, fmt;
    ICHECK(stream->Read(&code)) << "Loading code failed";
    ICHECK(stream->Read(&fmt)) << "Loading format failed";

    std::vector<std::string> tmp_func_names, tmp_const_vars;
    CHECK(stream->Read(&tmp_func_names)) << "Loading func names failed";
    CHECK(stream->Read(&tmp_const_vars)) << "Loading const vars failed";

    Array<String> func_names;
    for (auto func_name : tmp_func_names) func_names.push_back(String(func_name));

    Array<String> const_vars;
    for (auto const_var : tmp_const_vars) const_vars.push_back(String(const_var));

    auto n = make_object<CSourceModuleNode>(code, fmt, func_names, const_vars);
    return runtime::Module(n);
  }

  void SaveToFile(const String& file_name, const String& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "c" || fmt == "cc" || fmt == "cpp" || fmt == "cu") {
      ICHECK_NE(code_.length(), 0);
      SaveBinaryToFile(file_name, code_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    }
  }

  int GetPropertyMask() const override {
    return runtime::ModulePropertyMask::kBinarySerializable |
           runtime::ModulePropertyMask::kDSOExportable;
  }

  bool ImplementsFunction(const String& name, bool query_imports) final {
    return std::find(func_names_.begin(), func_names_.end(), name) != func_names_.end();
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

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_c")
    .set_body_typed(CSourceModuleNode::LoadFromBinary);

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
  const char* type_key() const final { return "c"; }

  String GetSource(const String& format) final { return code_.str(); }

  String GetFormat() override { return fmt_; }
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    return PackedFunc();
  }

  void SaveToFile(const String& file_name, const String& format) final {
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

  int GetPropertyMask() const override { return runtime::ModulePropertyMask::kDSOExportable; }

  bool ImplementsFunction(const String& name, bool query_imports) final {
    return std::find(func_names_.begin(), func_names_.end(), name) != func_names_.end();
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
      code_ << "(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, "
               "int* out_type_code, void* resource_handle);\n";
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

  void GenerateInternalBuffers() {
    if (metadata_->pool_inputs.defined()) {
      for (const auto& kv : metadata_->pool_inputs.value()) {
        tir::usmp::AllocatedPoolInfo allocated_pool_info = kv.second;
        if (allocated_pool_info->pool_info->is_internal) {
          if (const auto* pool_info = allocated_pool_info->pool_info.as<ConstantPoolInfoNode>()) {
            GenerateConstantBuffer(pool_info, allocated_pool_info->allocated_size->value);
          } else {
            GenerateWorkspaceBuffer(allocated_pool_info->pool_info.as<WorkspacePoolInfoNode>(),
                                    allocated_pool_info->allocated_size->value);
          }
        }
      }
    }
  }

  void GenerateIOWorkspaceMapFunction(const std::string& struct_type,
                                      const std::string& function_name,
                                      const Array<String>& tensor_names) {
    std::string map_function = runtime::get_name_mangled(metadata_->mod_name, function_name);
    code_ << "struct " << struct_type << " " << map_function << "(\n";
    std::string pools_struct = runtime::get_name_mangled(metadata_->mod_name, "workspace_pools");
    code_ << "  struct " << pools_struct << "* workspace_pools\n";
    code_ << "\n){\n";
    code_ << "struct " << struct_type << " ret = {\n";
    for (const String& name : tensor_names) {
      tir::usmp::PoolAllocation pool_allocation = metadata_->io_pool_allocations[name];
      code_ << "\t." << name << " = "
            << "&((uint8_t*)workspace_pools->" << pool_allocation->pool_info->pool_name << ")["
            << pool_allocation->byte_offset << "],\n";
    }
    code_ << "};\n";
    code_ << "return ret;\n";
    code_ << "}\n\n";
  }

  void GenerateConstantBuffer(const ConstantPoolInfoNode* pool_info, size_t allocated_size) {
    size_t ord = 0;
    if (pool_info->constant_info_array.size() > 0) {
      // Pool is RO, form an initialized struct
      code_ << "__attribute__((section(\".rodata.tvm\"), ";
      code_ << "))\n";
      code_ << "static const struct " << pool_info->pool_name << " {\n";
      // emit struct field names
      std::vector<ConstantInfo> const_info_vec(pool_info->constant_info_array.begin(),
                                               pool_info->constant_info_array.end());
      std::sort(const_info_vec.begin(), const_info_vec.end(),
                [](const ConstantInfo& a, const ConstantInfo& b) {
                  return a->byte_offset->value < b->byte_offset->value;
                });
      for (const auto& const_info : const_info_vec) {
        const auto& data = const_info->data;
        const auto& offs = const_info->byte_offset;
        int64_t num_elements = std::accumulate(data.Shape().begin(), data.Shape().end(), 1,
                                               std::multiplies<int64_t>());
        code_ << "  ";
        codegen_c_base_.PrintType(data.DataType(), code_);
        code_ << " " << const_info->name_hint << "[" << num_elements << "] __attribute__(("
              << (ord++ ? "packed, " : "") << "aligned(" << metadata_->constant_alignment << ")));";
        code_ << " // " << num_elements * data.DataType().bytes()
              << " bytes, aligned offset: " << offs << "\n";
      }
      code_ << "} " << pool_info->pool_name << " = {\n";

      // emit struct field initialization data
      for (const auto& const_info : const_info_vec) {
        code_ << "  ." << const_info->name_hint << " = {\n";
        codegen::NDArrayDataToC(const_info->data, 4, code_);
        code_ << "  },\n";
      }
      code_ << "};";
      code_ << "// of total size " << allocated_size << " bytes\n";
    } else {
      LOG(FATAL) << "No constant data in constant pool found " << GetRef<ObjectRef>(pool_info);
    }
  }

  void GenerateWorkspaceBuffer(const WorkspacePoolInfoNode* pool_info, size_t allocated_size) {
    code_ << "__attribute__((section(\".bss.noinit.tvm\"), ";
    code_ << "aligned(" << metadata_->workspace_alignment << ")))\n";
    code_ << "static uint8_t " << pool_info->pool_name << "[";
    code_ << allocated_size << "];\n";
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
      if (metadata_->io_pool_allocations.empty()) {
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
      if (metadata_->io_pool_allocations.empty()) {
        for (unsigned int i = 0; i < metadata_->inputs.size(); ++i) {
          call_args_ss << "((DLTensor*)(((TVMValue*)args)[" << i << "].v_handle))[0].data,";
        }
        for (unsigned int i = 0; i < metadata_->outputs.size(); ++i) {
          int j = metadata_->inputs.size() + i;
          call_args_ss << "((DLTensor*)(((TVMValue*)args)[" << j << "].v_handle))[0].data,";
        }
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

    if (metadata_->io_pool_allocations.empty()) {
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
    code_ << "(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* "
             "out_type_code, void* resource_handle);\n\n";

    code_ << "int32_t " << entrypoint_name;
    code_ << "(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* "
             "out_type_code, void* resource_handle) {\n";

    // We are creating a copy of the set of pointers
    size_t number_of_io_tensors = metadata_->inputs.size() + metadata_->outputs.size() +
                                  metadata_->pools.size() - metadata_->io_pool_allocations.size();
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
          code_ << "tensors[" << i << "] = ((TVMValue*)args)[" << run_func_to_entry_point_args[i]
                << "];\n";
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
    if (!metadata_->io_pool_allocations.empty()) {
      const std::string input_struct_type =
          runtime::get_name_mangled(metadata_->mod_name, "inputs");
      Array<String> input_tensor_names;
      for (const tir::Var& input_var : metadata_->inputs) {
        input_tensor_names.push_back(input_var->name_hint);
      }
      GenerateIOWorkspaceMapFunction(input_struct_type, "map_inputs", input_tensor_names);
      const std::string output_struct_type =
          runtime::get_name_mangled(metadata_->mod_name, "outputs");
      GenerateIOWorkspaceMapFunction(output_struct_type, "map_outputs", metadata_->outputs);
    }
    code_ << "TVM_DLL int32_t " << run_func << "(";
    {
      std::stringstream call_args_ss;
      if (metadata_->io_pool_allocations.empty()) {
        for (const tir::Var& input_var : metadata_->inputs) {
          if (input_var->type_annotation.defined()) {
            codegen_c_base_.PrintType(input_var->type_annotation, call_args_ss);
          } else {
            codegen_c_base_.PrintType(input_var.dtype(), call_args_ss);
          }
          call_args_ss << " " << tvm::runtime::SanitizeName(input_var->name_hint) << ",";
        }
        for (unsigned int i = 0; i < metadata_->outputs.size(); ++i) {
          call_args_ss << "void* output" << i << ",";
        }
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
      if (metadata_->io_pool_allocations.empty()) {
        call_args_ss << "struct " << runtime::get_name_mangled(mod_name, "inputs") << "* inputs,";
        call_args_ss << "struct " << runtime::get_name_mangled(mod_name, "outputs") << "* outputs,";
      }
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
      if (metadata_->io_pool_allocations.empty()) {
        for (const auto& input : metadata_->inputs) {
          call_args_ss << "inputs->" << tvm::runtime::SanitizeName(input->name_hint) << ",";
        }
        for (const auto& output : metadata_->outputs) {
          call_args_ss << "outputs->" << tvm::runtime::SanitizeName(output);
          call_args_ss << ",";
        }
      }

      for (const tir::Var& pool_var : metadata_->pools) {
        call_args_ss << "((uint8_t*)";
        String pool_name = metadata_->pool_inputs.value()[pool_var]->pool_info->pool_name;
        if (IsInternalWorkspaceBuffer(pool_var)) {
          call_args_ss << "&" << pool_name;
        } else {
          call_args_ss << "workspace_pools->" << tvm::runtime::SanitizeName(pool_name);
        }
        call_args_ss << "),";
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

    GenerateInternalBuffers();

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

class MetadataSerializer : public AttrVisitor {
 public:
  static constexpr const char* kGlobalSymbol = "kTvmgenMetadata";
  using MetadataKind = ::tvm::runtime::metadata::MetadataKind;

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

  // Serialiding NDArray as tuple of len, data
  void Visit(const char* key, runtime::NDArray* value) final {
    WriteComma();
    std::string bytes;
    dmlc::MemoryStringStream stream(&bytes);
    value->Save(&stream);
    // Serializing length of the data of NDArray
    code_ << stream.Tell();
    WriteComma();
    // Serializing NDArray as bytestream
    code_ << "\"";
    std::stringstream ss;
    char buf[6] = {0};
    for (uint8_t c : bytes) {
      snprintf(buf, sizeof(buf), "\\x%02x", c);
      ss << buf;
    }
    std::string as_bytes(ss.str());
    code_ << as_bytes;
    code_ << "\"\n";
  }

  void VisitArray(runtime::metadata::MetadataArray array) {
    auto old_is_first_item = is_first_item_;
    is_first_item_ = true;
    for (unsigned int i = 0; i < array->array.size(); ++i) {
      ObjectRef o = array->array[i];

      switch (array->kind) {
        case MetadataKind::kUint64: {
          int64_t i = Downcast<Integer>(o).IntValue();
          CHECK_GT(i, 0)
              << "Metadata is of type uint64_t, but array type contains a negative number";
          uint64_t ui = static_cast<uint64_t>(i);
          Visit(nullptr, &ui);
          continue;
        }
        case MetadataKind::kInt64: {
          int64_t i = Downcast<Integer>(o).IntValue();
          Visit(nullptr, &i);
          continue;
        }
        case MetadataKind::kBool: {
          bool b = Downcast<Bool>(o);
          Visit(nullptr, &b);
          break;
        }
        case MetadataKind::kString: {
          std::string s = Downcast<String>(o);
          Visit(nullptr, &s);
          break;
        }
        case MetadataKind::kHandle:
          CHECK(false) << "Don't know how to serialize handle";
          break;

        case MetadataKind::kMetadata: {
          runtime::metadata::MetadataBase metadata = Downcast<runtime::metadata::MetadataBase>(o);
          std::stringstream i_str;
          i_str << i;
          address_.push_back(i_str.str());
          Visit(nullptr, &metadata);
          address_.pop_back();
          break;
        }
        default:
          CHECK(false) << "Unknown MetadataKind for array: " << array->kind;
          break;
      }
      is_first_item_ = false;
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
      code_ << metadata::AddressFromParts(address_);
      if (key != nullptr) {
        address_.pop_back();
      }
      return;
    }

    runtime::metadata::MetadataBase metadata = Downcast<runtime::metadata::MetadataBase>(*value);
    if (key != nullptr) {  // NOTE: outermost call passes nullptr key
      address_.push_back(key);
    }
    WriteComma();
    code_ << "{\n";
    is_first_item_ = true;
    ReflectionVTable::Global()->VisitAttrs(metadata.operator->(), this);
    code_ << "}\n";
    if (key != nullptr) {  // NOTE: outermost call passes nullptr key
      address_.pop_back();
    }
  }

 private:
  void EmitCType(const runtime::metadata::MetadataArrayNode* arr, std::ostream& os) {
    switch (arr->kind) {
      case MetadataKind::kUint64:
        os << "uint64_t";
        break;
      case MetadataKind::kInt64:
        os << "int64_t";
        break;
      case MetadataKind::kBool:
        os << "bool";
        break;
      case MetadataKind::kString:
        os << "const char*";
        break;
      case MetadataKind::kHandle:
        os << "void*";
        break;
      case MetadataKind::kMetadata:
        os << "struct " << arr->get_element_c_struct_name();
        break;
      default:
        CHECK(false) << "Unknown kind in MetadataArray: " << arr->kind
                     << " (struct_name=" << arr->get_c_struct_name() << ")";
        break;
    }
  }

 public:
  void CodegenMetadata(::tvm::runtime::metadata::Metadata metadata) {
    decl_ << "#include <inttypes.h>" << std::endl
          << "#include <tvm/runtime/metadata_types.h>" << std::endl
          << "#include <tvm/runtime/c_runtime_api.h>" << std::endl;
    std::vector<metadata::DiscoverArraysVisitor::DiscoveredArray> queue;
    metadata::DiscoverArraysVisitor array_discover{&queue};
    array_discover.Visit(metadata::kMetadataGlobalSymbol, &metadata);

    for (auto item : queue) {
      auto struct_address = std::get<0>(item);
      address_.push_back(struct_address);

      auto arr = std::get<1>(item);

      // Prepend const with everything except C-string, which needs appending.
      code_ << "static ";
      if (arr->kind != MetadataKind::kString) {
        code_ << "const ";
      }
      EmitCType(arr.operator->(), code_);
      if (arr->kind == MetadataKind::kString) {
        code_ << " const";
      }
      code_ << " " << struct_address << "[" << arr->array.size() << "] = {" << std::endl;
      is_first_item_ = true;

      VisitArray(arr);
      address_.pop_back();
      code_ << "};" << std::endl;
    }

    // Finally, emit overall struct.
    address_.push_back(metadata::kMetadataGlobalSymbol);
    code_ << "static const struct TVMMetadata " << metadata::AddressFromParts(address_) << "[1] = {"
          << std::endl;
    Visit(nullptr, &metadata);
    code_ << "};" << std::endl;
    address_.pop_back();
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

namespace {
runtime::Module CreateAotMetadataModule(runtime::metadata::Metadata aot_metadata,
                                        bool is_c_runtime) {
  MetadataSerializer serializer;
  serializer.CodegenMetadata(aot_metadata);
  std::stringstream lookup_func;
  std::string get_c_metadata_func_name;

  // NOTE: mangling is not needed in the c++ runtime because the function
  //       name is looked-up via LibraryModule.
  // TODO(alanmacd): unify these two approaches

  if (is_c_runtime == true) {
    get_c_metadata_func_name = runtime::get_name_mangled(
        aot_metadata->mod_name(), ::tvm::runtime::symbol::tvm_get_c_metadata);
  } else {
    get_c_metadata_func_name = ::tvm::runtime::symbol::tvm_get_c_metadata;
  }

  lookup_func << "#ifdef __cplusplus\n"
              << "extern \"C\"\n"
              << "#endif\n";

  lookup_func << "TVM_DLL int32_t " << get_c_metadata_func_name
              << "(TVMValue* arg_values, int* arg_tcodes, int "
                 "num_args, TVMValue* ret_values, int* ret_tcodes, void* resource_handle) {"
              << std::endl;
  lookup_func << "    ret_values[0].v_handle = (void*) &" << MetadataSerializer::kGlobalSymbol
              << ";" << std::endl;
  lookup_func << "    ret_tcodes[0] = kTVMOpaqueHandle;" << std::endl;
  lookup_func << "    return 0;" << std::endl;
  lookup_func << "};" << std::endl;
  std::vector<String> func_names{get_c_metadata_func_name};
  return CSourceModuleCreate(serializer.GetOutput() + lookup_func.str(), "c", func_names,
                             Array<String>());
}
}  // namespace

runtime::Module CreateCSourceCrtMetadataModule(const Array<runtime::Module>& modules, Target target,
                                               relay::Runtime runtime,
                                               relay::backend::ExecutorCodegenMetadata metadata,
                                               runtime::metadata::Metadata aot_metadata) {
  Array<runtime::Module> final_modules(modules);
  Array<String> func_names;

  if (metadata.defined()) {
    if (metadata->executor == "aot") {
      if (aot_metadata.defined()) {
        final_modules.push_back(CreateAotMetadataModule(aot_metadata, true));
      }

      // add the run function (typically "tvmgen_default_run") to function registry
      // when using AOT executor
      std::string run_func = runtime::get_name_mangled(metadata->mod_name, "run");
      func_names.push_back(run_func);
    }
  }

  for (runtime::Module mod : final_modules) {
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
  for (const auto& mod : final_modules) {
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
  lookup_func << "    ret_values[0].v_handle = (void*) &" << metadata::kMetadataGlobalSymbol << ";"
              << std::endl;
  lookup_func << "    ret_tcodes[0] = kTVMOpaqueHandle;" << std::endl;
  lookup_func << "    return 0;" << std::endl;
  lookup_func << "};" << std::endl;

  auto mod = MetadataModuleCreate(metadata);
  mod->Import(CreateAotMetadataModule(metadata, false));
  return mod;
}

// supports limited save without cross compile
class DeviceSourceModuleNode final : public runtime::ModuleNode {
 public:
  DeviceSourceModuleNode(std::string data, std::string fmt,
                         std::unordered_map<std::string, FunctionInfo> fmap, std::string type_key,
                         std::function<std::string(const std::string&)> fget_source)
      : data_(data), fmt_(fmt), fmap_(fmap), type_key_(type_key), fget_source_(fget_source) {}

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  String GetSource(const String& format) final {
    if (fget_source_ != nullptr) {
      return fget_source_(format);
    } else {
      return data_;
    }
  }

  const char* type_key() const final { return type_key_.c_str(); }
  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return runtime::ModulePropertyMask::kBinarySerializable; }

  void SaveToFile(const String& file_name, const String& format) final {
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
                                            relay::backend::ExecutorCodegenMetadata(),
                                            runtime::metadata::Metadata());
    });

}  // namespace codegen
}  // namespace tvm

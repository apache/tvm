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

#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../runtime/file_utils.h"
#include "codegen_source_base.h"

namespace tvm {
namespace codegen {

using ffi::Any;
using ffi::Function;
using ffi::PackedArgs;

using runtime::FunctionInfo;
using runtime::GetFileFormat;
using runtime::GetMetaFilePath;
using runtime::SaveBinaryToFile;

// Simulator function
class SourceModuleNode : public runtime::ModuleNode {
 public:
  SourceModuleNode(std::string code, std::string fmt) : code_(code), fmt_(fmt) {}
  const char* type_key() const final { return "source"; }

  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return ffi::Function();
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

  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    // Currently c-source module is used as demonstration purposes with binary metadata module
    // that expects get_symbol interface. When c-source module is used as external module, it
    // will only contain one function. However, when its used as an internal module (e.g., target
    // "c") it can have many functions.
    if (name == "get_symbol") {
      return ffi::Function(
          [sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) { *rv = this->func_names_[0]; });
    } else if (name == "get_const_vars") {
      return ffi::Function(
          [sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) { *rv = this->const_vars_; });
    } else if (name == "get_func_names") {
      return ffi::Function(
          [sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) { *rv = this->func_names_; });
    } else {
      return ffi::Function(nullptr);
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

TVM_FFI_REGISTER_GLOBAL("runtime.module.loadbinary_c")
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

// supports limited save without cross compile
class DeviceSourceModuleNode final : public runtime::ModuleNode {
 public:
  DeviceSourceModuleNode(std::string data, std::string fmt,
                         std::unordered_map<std::string, FunctionInfo> fmap, std::string type_key,
                         std::function<std::string(const std::string&)> fget_source)
      : data_(data), fmt_(fmt), fmap_(fmap), type_key_(type_key), fget_source_(fget_source) {}

  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return ffi::Function();
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

TVM_FFI_REGISTER_GLOBAL("runtime.SourceModuleCreate").set_body_typed(SourceModuleCreate);

TVM_FFI_REGISTER_GLOBAL("runtime.CSourceModuleCreate")
    .set_body_typed([](String code, String fmt, Optional<Array<String>> func_names,
                       Optional<Array<String>> const_vars) {
      return CSourceModuleCreate(code, fmt, func_names.value_or({}), const_vars.value_or({}));
    });

}  // namespace codegen
}  // namespace tvm

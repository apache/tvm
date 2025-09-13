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
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

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
class SourceModuleNode : public ffi::ModuleObj {
 public:
  SourceModuleNode(std::string code, std::string fmt) : code_(code), fmt_(fmt) {}
  const char* kind() const final { return "source"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
  }

  ffi::String InspectSource(const ffi::String& format) const final { return code_; }

  ffi::Array<ffi::String> GetWriteFormats() const override { return {fmt_}; }

 protected:
  std::string code_;
  std::string fmt_;
};

ffi::Module SourceModuleCreate(std::string code, std::string fmt) {
  auto n = ffi::make_object<SourceModuleNode>(code, fmt);
  return ffi::Module(n);
}

// Simulator function
class CSourceModuleNode : public ffi::ModuleObj {
 public:
  CSourceModuleNode(const std::string& code, const std::string& fmt,
                    const ffi::Array<ffi::String>& func_names,
                    const ffi::Array<ffi::String>& const_vars)
      : code_(code), fmt_(fmt), const_vars_(const_vars), func_names_(func_names) {
    if (fmt_.empty()) fmt_ = "c";
  }

  const char* kind() const final { return "c"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);
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

  ffi::String InspectSource(const ffi::String& format) const final { return code_; }

  ffi::Array<ffi::String> GetWriteFormats() const override { return {fmt_}; }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    dmlc::MemoryStringStream ms(&buffer);
    dmlc::Stream* stream = &ms;
    stream->Write(code_);
    stream->Write(fmt_);

    std::vector<std::string> func_names;
    for (const auto func_name : func_names_) func_names.push_back(func_name);
    std::vector<std::string> const_vars;
    for (auto const_var : const_vars_) const_vars.push_back(const_var);
    stream->Write(func_names);
    stream->Write(const_vars);
    return ffi::Bytes(buffer);
  }

  static ffi::Module LoadFromBytes(const ffi::Bytes& bytes) {
    dmlc::MemoryFixedSizeStream ms(const_cast<char*>(bytes.data()), bytes.size());
    dmlc::Stream* stream = &ms;

    std::string code, fmt;
    ICHECK(stream->Read(&code)) << "Loading code failed";
    ICHECK(stream->Read(&fmt)) << "Loading format failed";

    std::vector<std::string> tmp_func_names, tmp_const_vars;
    CHECK(stream->Read(&tmp_func_names)) << "Loading func names failed";
    CHECK(stream->Read(&tmp_const_vars)) << "Loading const vars failed";

    ffi::Array<ffi::String> func_names;
    for (auto func_name : tmp_func_names) func_names.push_back(ffi::String(func_name));

    ffi::Array<ffi::String> const_vars;
    for (auto const_var : tmp_const_vars) const_vars.push_back(ffi::String(const_var));

    auto n = ffi::make_object<CSourceModuleNode>(code, fmt, func_names, const_vars);
    return ffi::Module(n);
  }

  void WriteToFile(const ffi::String& file_name, const ffi::String& format) const final {
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
    return ffi::Module::kBinarySerializable | ffi::Module::kCompilationExportable;
  }

  bool ImplementsFunction(const ffi::String& name) final {
    return std::find(func_names_.begin(), func_names_.end(), name) != func_names_.end();
  }

 protected:
  std::string code_;
  std::string fmt_;
  ffi::Array<ffi::String> const_vars_;
  ffi::Array<ffi::String> func_names_;
};

ffi::Module CSourceModuleCreate(const ffi::String& code, const ffi::String& fmt,
                                const ffi::Array<ffi::String>& func_names,
                                const ffi::Array<ffi::String>& const_vars) {
  auto n = ffi::make_object<CSourceModuleNode>(code.operator std::string(),
                                               fmt.operator std::string(), func_names, const_vars);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.Module.load_from_bytes.c", CSourceModuleNode::LoadFromBytes);
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

// supports limited save without cross compile
class DeviceSourceModuleNode final : public ffi::ModuleObj {
 public:
  DeviceSourceModuleNode(std::string data, std::string fmt,
                         std::unordered_map<std::string, FunctionInfo> fmap, std::string type_key,
                         std::function<std::string(const std::string&)> fget_source)
      : data_(data), fmt_(fmt), fmap_(fmap), type_key_(type_key), fget_source_(fget_source) {}

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (fget_source_ != nullptr) {
      return fget_source_(format);
    } else {
      return data_;
    }
  }

  const char* kind() const final { return type_key_.c_str(); }
  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; }

  void WriteToFile(const ffi::String& file_name, const ffi::String& format) const final {
    std::string fmt = GetFileFormat(file_name, format);
    ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    std::string meta_file = GetMetaFilePath(file_name);
    SaveMetaDataToFile(meta_file, fmap_);
    SaveBinaryToFile(file_name, data_);
  }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    dmlc::MemoryStringStream ms(&buffer);
    dmlc::Stream* stream = &ms;
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
    return ffi::Bytes(buffer);
  }

 private:
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
  std::string type_key_;
  std::function<std::string(const std::string&)> fget_source_;
};

ffi::Module DeviceSourceModuleCreate(std::string data, std::string fmt,
                                     std::unordered_map<std::string, FunctionInfo> fmap,
                                     std::string type_key,
                                     std::function<std::string(const std::string&)> fget_source) {
  auto n = ffi::make_object<DeviceSourceModuleNode>(data, fmt, fmap, type_key, fget_source);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.SourceModuleCreate", SourceModuleCreate)
      .def("runtime.CSourceModuleCreate", [](ffi::String code, ffi::String fmt,
                                             ffi::Optional<ffi::Array<ffi::String>> func_names,
                                             ffi::Optional<ffi::Array<ffi::String>> const_vars) {
        return CSourceModuleCreate(code, fmt, func_names.value_or({}), const_vars.value_or({}));
      });
}

}  // namespace codegen
}  // namespace tvm

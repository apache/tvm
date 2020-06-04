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
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "../../runtime/file_util.h"
#include "../../runtime/meta_data.h"
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
  CSourceModuleNode(std::string code, std::string fmt) : code_(code), fmt_(fmt) {}
  const char* type_key() const { return "c"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "C Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  std::string GetSource(const std::string& format) final { return code_; }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "cc") {
      CHECK_NE(code_.length(), 0);
      SaveBinaryToFile(file_name, code_);
    } else {
      CHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    }
  }

 protected:
  std::string code_;
  std::string fmt_;
};

runtime::Module CSourceModuleCreate(std::string code, std::string fmt) {
  auto n = make_object<CSourceModuleNode>(code, fmt);
  return runtime::Module(n);
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
    CHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
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

// Pack the source code and metadata, where source code could be any
// user-defined code, i.e. c source code, json graph representation, etc.
class PackagingModule final : public runtime::ModuleNode {
 public:
  PackagingModule(Map<String, String> code, const std::string& fmt,
                  Map<String, Map<String, runtime::NDArray>> metadata)
      : code_(code), fmt_(fmt), metadata_(metadata) {}

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_source") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetSource(); });
    } else if (name == "get_metadata") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetMetadata(); });
    } else if (name == "is_c_source") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->IsCSourceCode(); });
    } else {
      LOG(FATAL) << "Unknown packed function: " << name;
      return PackedFunc(nullptr);
    }
  }

  Map<String, String> GetSource() const { return code_; }

  Map<String, Map<String, runtime::NDArray>> GetMetadata() const { return metadata_; }

  bool IsCSourceCode() { return fmt_ == "c" || fmt_ == "cc"; }

  const char* type_key() const { return "c"; }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    CHECK_EQ(fmt, "cc") << "file_name: " << file_name << " must be a .cc file.";
    SaveBinaryToFile(file_name, ";");
  }

 private:
  /*! \brief Symbol to source (e.g. c source/json) mapping. */
  Map<String, String> code_;
  std::string fmt_;
  /*! \brief Symbol to {var_name : NDArray} pair mapping. */
  Map<String, Map<String, runtime::NDArray>> metadata_;
};

runtime::Module PackagingModuleCreate(Map<String, String> code, std::string fmt,
                                      Map<String, Map<String, runtime::NDArray>> metadata) {
  auto n = make_object<PackagingModule>(code, fmt, metadata);
  return runtime::Module(n);
}

class ModuleInitWrapper : public runtime::ModuleNode {
 public:
  ModuleInitWrapper(Map<String, Map<String, runtime::NDArray>> metadata, Map<String, String> code)
      : metadata_(metadata), code_(code) {}

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (initialized_.count(name) == 0) {
      this->InitSubModule(name);
      initialized_[name] = true;
    } else if (name != "__InitModule" && name != "__DestroyModule") {
      CHECK(!this->imports().empty());
      runtime::Module submodule = this->imports().at(0);
      return submodule->GetFunction(name);
    }

    return PackedFunc();
  }
  
  const char* type_key() const { return "module_init"; }

  void InitSubModule(const std::string& symbol) {}

 private:
  std::unordered_map<std::string, bool> initialized_;
  /*! \brief A symbol to {var_name : NDArray} pair mapping. */
  Map<String, Map<String, runtime::NDArray>> metadata_;
  /*!
   * \brief For JSON runtime we need the json code to build up an engine. For
   * c source module, code has already been compiled into a DSO module, only
   * metadata is needed to feed the correct data.
   */
  Map<String, String> code_;
};

runtime::Module ModuleInitWrapperCreate(Map<String, Map<String, runtime::NDArray>> metadata,
                                        Map<String, String> code) {
  auto n = make_object<ModuleInitWrapper>(metadata, code);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.PackagingModuleCreate").set_body_typed(PackagingModuleCreate);

TVM_REGISTER_GLOBAL("runtime.SourceModuleCreate").set_body_typed(SourceModuleCreate);

TVM_REGISTER_GLOBAL("runtime.CSourceModuleCreate").set_body_typed([](String code, String fmt) {
  return CSourceModuleCreate(code.operator std::string(), fmt.operator std::string());
});

TVM_REGISTER_GLOBAL("runtime.ModuleInitWrapper")
    .set_body_typed([](Map<String, Map<String, runtime::NDArray>> metadata,
                       Map<String, String> code) {
      return ModuleInitWrapperCreate(metadata, code);
    });

}  // namespace codegen
}  // namespace tvm

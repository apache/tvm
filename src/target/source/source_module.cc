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
class SourceMetadataModuleNode final : public runtime::ModuleNode {
 public:
  SourceMetadataModuleNode(Map<String, String> code, const std::string& source_type,
                           Map<String, Map<String, runtime::NDArray>> metadata)
      : code_(code), source_type_(source_type), metadata_(metadata) {}

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_source") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->code_; });
    } else if (name == "get_source_type") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->source_type_; });
    } else if (name == "get_metadata") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->metadata_; });
    } else {
      LOG(FATAL) << "Unknown packed function: " << name;
      return PackedFunc(nullptr);
    }
  }

  const char* type_key() const { return "c"; }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string source_type = GetFileFormat(file_name, format);
    CHECK_EQ(source_type, "cc") << "file_name: " << file_name << " must be a .cc file.";
    SaveBinaryToFile(file_name, ";");
  }

 private:
  /*! \brief Symbol to source (e.g. c source/json) mapping. */
  Map<String, String> code_;
  /*! \brief The type of the source code, e.g. c or any customized json type. */
  std::string source_type_;
  /*! \brief Symbol to {var_name : NDArray} pair mapping. */
  Map<String, Map<String, runtime::NDArray>> metadata_;
};

runtime::Module SourceMetadataModuleCreate(Map<String, String> code, std::string source_type,
                                           Map<String, Map<String, runtime::NDArray>> metadata) {
  auto n = make_object<SourceMetadataModuleNode>(code, source_type, metadata);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.SourceMetadataModuleCreate")
    .set_body_typed(SourceMetadataModuleCreate);

TVM_REGISTER_GLOBAL("runtime.SourceModuleCreate").set_body_typed(SourceModuleCreate);

TVM_REGISTER_GLOBAL("runtime.CSourceModuleCreate")
    .set_body_typed([](String code, String source_type) {
      return CSourceModuleCreate(code.operator std::string(), source_type.operator std::string());
    });

}  // namespace codegen
}  // namespace tvm

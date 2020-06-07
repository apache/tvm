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
#include <cstdint>

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
  PackagingModule(Map<String, String> code, const std::string& source_type,
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
    } else if (name == "is_c_source") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->IsCSourceCode(); });
    } else {
      LOG(FATAL) << "Unknown packed function: " << name;
      return PackedFunc(nullptr);
    }
  }

  bool IsCSourceCode() { return source_type_ == "c" || source_type_ == "cc"; }

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

runtime::Module PackagingModuleCreate(Map<String, String> code, std::string source_type,
                                      Map<String, Map<String, runtime::NDArray>> metadata) {
  auto n = make_object<PackagingModule>(code, source_type, metadata);
  return runtime::Module(n);
}

class ModuleInitWrapper : public runtime::ModuleNode {
 public:
  ModuleInitWrapper(Map<String, Map<String, runtime::NDArray>> metadata, String source_type)
      : metadata_(metadata), source_type_(source_type) {}

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (initialized_.count(name) == 0) {
      this->InitSubModule(name);
      initialized_[name] = true;
    }

    if (name != "init_module" && name != "destroy_module") {
      CHECK(!this->imports().empty());
      runtime::Module submodule = this->imports().at(0);
      return submodule->GetFunction(name);
    }

    return PackedFunc();
  }

  const char* type_key() const { return "module_init"; }

  template <typename T>
  std::string GetElements(const std::string& var_name, const std::string& type_name,
                          const runtime::NDArray& arr) {
    std::ostringstream os;
    os.precision(16);
    // Get the number of elements.
    int64_t num_elems = 1;
    for (auto i : arr.Shape()) num_elems *= i;
    os << "static " << type_name << " " << var_name << "[" << num_elems << "] = {";
    T* ptr = static_cast<T*>(arr->data);
    for (int64_t i = 0; i < num_elems - 1; i++) {
      os << ptr[i] << ",";
    }
    if (num_elems > 0) os << ptr[num_elems - 1];
    os << "};\n";
    return os.str();
  }

  std::string InitCSourceMetadata() {
    std::string ret = "";
    for (const auto& it : metadata_) {
      for (const auto& vars : it.second) {
        std::string var_name = vars.first.operator std::string();
        runtime::NDArray data = vars.second;
        CHECK(data->dtype.lanes == 1);
        if (data->dtype.code == kDLFloat) {
          if (data->dtype.bits == 32) {
            ret += GetElements<float>(var_name, "float", data);
          } else {
            CHECK_EQ(data->dtype.bits, 64);
            ret += GetElements<double>(var_name, "double", data);
          }
        } else if (data->dtype.code == kDLUInt) {
          if (data->dtype.bits == 8) {
            ret += GetElements<uint8_t>(var_name, "uint8_t", data);
          } else {
            CHECK_EQ(data->dtype.bits, 32);
            ret += GetElements<uint32_t>(var_name, "uint32_t", data);
          }
        } else {
          if (data->dtype.bits == 8) {
            ret += GetElements<int8_t>(var_name, "int8_t", data);
          } else {
            CHECK_EQ(data->dtype.bits, 32);
            ret += GetElements<int32_t>(var_name, "int32_t", data);
          }
        }
      }
    }
    return ret;
  }

  void InitSubModule(const std::string& symbol) {
    // Dispatch initializer according to the source type
    // std::string initializer = "runtime.init." + source_type_;
    // auto pf = tvm::runtime::Registry::Get(initializer);

    // CHECK(pf) << "Failed to find the initializer for " << initializer;
    if (source_type_ != "c") {
      LOG(FATAL) << "Implement the initialization of json style runtime here";
    }
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    if (source_type_ == "c") {
      std::string fmt = GetFileFormat(file_name, format);
      CHECK_EQ(fmt, "h") << "Can only save to .h file";
      SaveBinaryToFile(file_name, InitCSourceMetadata());
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(source_type_.operator std::string());

    // Save the total number of symbols
    uint64_t sym_cnt = static_cast<uint64_t>(metadata_.size());
    stream->Write(sym_cnt);

    for (const auto& it : metadata_) {
      // Save the symbol/function name
      stream->Write(it.first.operator std::string());

      std::vector<std::string> variables;
      std::vector<runtime::NDArray> metadata;
      for (const auto& vit : it.second) {
        String var_name = vit.first;
        variables.push_back(var_name.operator std::string());
        metadata.push_back(vit.second);
      }

      // Save all variables in the function.
      stream->Write(variables);
      // Save all constant data
      uint64_t sz = static_cast<uint64_t>(metadata.size());
      stream->Write(sz);
      for (uint64_t i = 0; i < sz; i++) {
        metadata[i].Save(stream);
      }
    }
  }

  static runtime::Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string source_type;
    CHECK(stream->Read(&source_type)) << "Loading source type failed";

    Map<String, Map<String, runtime::NDArray>> metadata;

    uint64_t sym_cnt;
    CHECK(stream->Read(&sym_cnt, sizeof(sym_cnt))) << "Loading the number of symbols failed";

    for (uint64_t i = 0; i < sym_cnt; i++) {
      std::string sym;
      CHECK(stream->Read(&sym)) << "Loading symbol name failed";
      // Load variable and ndarray pairs
      std::vector<std::string> variables;
      std::vector<runtime::NDArray> arrays;
      CHECK(stream->Read(&variables)) << "Loading variables failed";
      uint64_t sz;
      CHECK(stream->Read(&sz, sizeof(sz))) << "Loading medata size failed";
      CHECK_EQ(static_cast<size_t>(sz), variables.size())
          << "The number of variables and ndarray counts must match";
      for (uint64_t i = 0; i < sz; i++) {
        tvm::runtime::NDArray temp;
        temp.Load(stream);
        arrays.push_back(temp);
      }
      Map<String, runtime::NDArray> var_const;
      for (size_t i = 0; i < variables.size(); i++) {
        var_const.Set(variables[i], arrays[i]);
      }
      metadata.Set(sym, var_const);
    }
    auto n = runtime::make_object<ModuleInitWrapper>(metadata, source_type);
    return runtime::Module(n);
  }

 private:
  std::unordered_map<std::string, bool> initialized_;
  /*! \brief A symbol to {var_name : NDArray} pair mapping. */
  Map<String, Map<String, runtime::NDArray>> metadata_;
  /*! \brief The type of the source, i.e. c, or any customized json */
  String source_type_;
};

runtime::Module ModuleInitWrapperCreate(Map<String, Map<String, runtime::NDArray>> metadata,
                                        String source_type) {
  auto n = make_object<ModuleInitWrapper>(metadata, source_type);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.PackagingModuleCreate").set_body_typed(PackagingModuleCreate);

TVM_REGISTER_GLOBAL("runtime.SourceModuleCreate").set_body_typed(SourceModuleCreate);

TVM_REGISTER_GLOBAL("runtime.CSourceModuleCreate")
    .set_body_typed([](String code, String source_type) {
      return CSourceModuleCreate(code.operator std::string(), source_type.operator std::string());
    });

TVM_REGISTER_GLOBAL("runtime.ModuleInitWrapper")
    .set_body_typed([](Map<String, Map<String, runtime::NDArray>> metadata, String source_type) {
      return ModuleInitWrapperCreate(metadata, source_type);
    });

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_module_init")
    .set_body_typed(ModuleInitWrapper::LoadFromBinary);
}  // namespace codegen
}  // namespace tvm

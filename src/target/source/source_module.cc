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
#include <tvm/runtime/metadata.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/name_transforms.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../runtime/file_utils.h"
#include "../metadata_utils.h"
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

}  // namespace codegen
}  // namespace tvm
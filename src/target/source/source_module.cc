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

/*!
 * \brief Create a metadata module wrapper. The helper is used by different
 *        codegens, such as graph runtime codegen and the vm compiler.
 *
 * \param params The metadata for initialization of all modules.
 * \param dso_module The DSO module that contains TVM primitives.
 * \param modules The submodules that will be wrapped, e.g. CSource modules that
 *        contain vendor library calls or customized runtime modules.
 *
 * \return The created metadata module that manages initialization of metadata.
 */
runtime::Module CreateMetadataModule(
    const std::unordered_map<std::string, runtime::NDArray>& params,
    const runtime::Module& dso_module, const Array<runtime::Module>& modules) {
  // Wrap all submodules in the initialization wrapper.
  std::unordered_map<std::string, std::vector<std::string>> sym_metadata;
  for (runtime::Module it : modules) {
    auto pf_sym = it.GetFunction("get_symbol");
    auto pf_var = it.GetFunction("get_const_vars");
    if (pf_sym != nullptr && pf_var != nullptr) {
      String symbol = pf_sym();
      Array<String> variables = pf_var();
      std::vector<std::string> arrays;
      for (size_t i = 0; i < variables.size(); i++) {
        arrays.push_back(variables[i].operator std::string());
      }
      CHECK_EQ(sym_metadata.count(symbol), 0U) << "Found duplicated symbol: " << symbol;
      sym_metadata[symbol] = arrays;
    }
  }

  // Wrap the modules.
  runtime::Module init_m = runtime::MetadataModuleCreate(params, sym_metadata);
  init_m.Import(dso_module);
  for (const auto& it : modules) {
    init_m.Import(it);
  }

  return init_m;
}

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
  CSourceModuleNode(const std::string& code, const std::string& fmt, const std::string& symbol,
                    const Array<String>& const_vars)
      : code_(code), fmt_(fmt), symbol_(symbol), const_vars_(const_vars) {}
  const char* type_key() const { return "c"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_; });
    } else if (name == "get_const_vars") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->const_vars_; });
    } else {
      return PackedFunc(nullptr);
    }
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
  std::string symbol_;
  Array<String> const_vars_;
};

runtime::Module CSourceModuleCreate(const String& code, const String& fmt, const String& symbol,
                                    const Array<String>& const_vars) {
  auto n = make_object<CSourceModuleNode>(code.operator std::string(), fmt.operator std::string(),
                                          symbol.operator std::string(), const_vars);
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

TVM_REGISTER_GLOBAL("runtime.SourceModuleCreate").set_body_typed(SourceModuleCreate);

TVM_REGISTER_GLOBAL("runtime.CSourceModuleCreate")
    .set_body_typed([](String code, String fmt, String symbol, Array<String> const_vars) {
      return CSourceModuleCreate(code, fmt, symbol, const_vars);
    });

}  // namespace codegen
}  // namespace tvm

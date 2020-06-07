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
 * \file src/runtime/module_init_wrapper.cc
 * \brief A wrapper for initializing modules using metadata
 */
#include <tvm/node/container.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <cstdint>
#include <sstream>

#include "file_util.h"

namespace tvm {
namespace runtime {

class CSourceMetadataInitializer {
 public:
  explicit CSourceMetadataInitializer(Map<String, Map<String, runtime::NDArray>> metadata)
      : metadata_(metadata) {}

  template <typename T>
  void GetElements(const std::string& var_name, const std::string& type_name,
                   const runtime::NDArray& arr) {
    // Get the number of elements.
    int64_t num_elems = 1;
    for (auto i : arr.Shape()) num_elems *= i;
    stream_ << "static " << type_name << " " << var_name << "[" << num_elems << "] = {";
    T* ptr = static_cast<T*>(arr->data);
    for (int64_t i = 0; i < num_elems - 1; i++) {
      stream_ << ptr[i] << ",";
    }
    if (num_elems > 0) stream_ << ptr[num_elems - 1];
    stream_ << "};\n";
  }

  std::string Init() {
    for (const auto& it : metadata_) {
      for (const auto& vars : it.second) {
        std::string var_name = vars.first.operator std::string();
        runtime::NDArray data = vars.second;
        CHECK(data->dtype.lanes == 1);
        if (data->dtype.code == kDLFloat) {
          if (data->dtype.bits == 32) {
            stream_.precision(std::numeric_limits<float>::digits10 + 1);
            GetElements<float>(var_name, "float", data);
          } else {
            CHECK_EQ(data->dtype.bits, 64);
            stream_.precision(std::numeric_limits<double>::digits10 + 1);
            GetElements<double>(var_name, "double", data);
          }
        } else if (data->dtype.code == kDLUInt) {
          if (data->dtype.bits == 8) {
            GetElements<uint8_t>(var_name, "uint8_t", data);
          } else {
            CHECK_EQ(data->dtype.bits, 32);
            GetElements<uint32_t>(var_name, "uint32_t", data);
          }
        } else {
          if (data->dtype.bits == 8) {
            GetElements<int8_t>(var_name, "int8_t", data);
          } else {
            CHECK_EQ(data->dtype.bits, 32);
            GetElements<int32_t>(var_name, "int32_t", data);
          }
        }
      }
    }
    return stream_.str();
  }

 private:
  /*! \brief The stream to print constant data. */
  std::ostringstream stream_;
  /*! \brief A symbol to {var_name : NDArray} pair mapping. */
  Map<String, Map<String, runtime::NDArray>> metadata_;
};

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

  void InitSubModule(const std::string& symbol) {
    // Dispatch initializer according to the source type
    if (source_type_ != "c") {
      LOG(FATAL) << "Implement the initialization of json style runtime here";
    } else {
      // TODO(zhiics) Handle json runtime.
      // std::string initializer = "runtime.init." + source_type_;
      // auto pf = tvm::runtime::Registry::Get(initializer);
      // CHECK(pf) << "Failed to find the initializer for " << initializer;
    }
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    // C source module relies on AOT compilation. The source code has already
    // been generated. The used metadata is saved a separate file for
    // compilation.
    if (source_type_ == "c") {
      std::string fmt = GetFileFormat(file_name, format);
      CHECK_EQ(fmt, "h") << "Can only save to .h file";
      CSourceMetadataInitializer c_init(metadata_);
      SaveBinaryToFile(file_name, c_init.Init());
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
  /*!
   * \brief Record if a module is initialized. It is needed by imported
   * modules using execution engine.
   */
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

TVM_REGISTER_GLOBAL("runtime.ModuleInitWrapper")
    .set_body_typed([](Map<String, Map<String, runtime::NDArray>> metadata, String source_type) {
      return ModuleInitWrapperCreate(metadata, source_type);
    });

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_module_init")
    .set_body_typed(ModuleInitWrapper::LoadFromBinary);
}  // namespace runtime
}  // namespace tvm

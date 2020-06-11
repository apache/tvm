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

using StringNDArrayMap = std::unordered_map<String, runtime::NDArray, ObjectHash, ObjectEqual>;

class CSourceMetadataInitializer {
 public:
  explicit CSourceMetadataInitializer(const StringNDArrayMap& metadata) : metadata_(metadata) {}

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
      std::string var_name = it.first.operator std::string();
      runtime::NDArray data = it.second;
      CHECK_EQ(data->dtype.lanes, 1U);
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
    return stream_.str();
  }

 private:
  /*! \brief The stream to print constant data. */
  std::ostringstream stream_;
  /*! \brief variable name to NDArray mapping. */
  StringNDArrayMap metadata_;
};

class ModuleInitWrapper : public runtime::ModuleNode {
 public:
  ModuleInitWrapper(const Array<String>& variables, const Array<runtime::NDArray>& metadata) {
    CHECK_EQ(variables.size(), metadata.size());
    for (size_t i = 0; i < variables.size(); i++) {
      metadata_[variables[i]] = metadata[i];
    }
  }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (initialized_.count(name) == 0) {
      this->InitSubModule(name);
      initialized_[name] = true;
    }

    if (name != "__InitModule") {
      CHECK(!this->imports().empty());
      runtime::Module submodule = this->imports().at(0);
      return submodule->GetFunction(name);
    }

    return PackedFunc();
  }

  const char* type_key() const { return "module_init"; }

  /*!
   * \brief Initialize each imported module.
   * \param symobl The symbol used for initializing a module. It is also used
   * for runtime lookup.
   *
   * \note  A module could be like the following:
   *  ModuleInitWrapper (contains all the metadata)
   *    - CSourceModule
   *    - JSON runtime module
   *
   *  The initializer iterates through the wrapped module and intilizes them
   *  accordingly by passing the needed metadata into it.
   */
  void InitSubModule(const std::string& symbol) {
    // Dispatch initializer according to the source type
    // TODO(zhiics) iterate through the imported modules to initialize
    // for (const auto& it : this->imports()) {
    // }
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    // C source module relies on AOT compilation. The source code has already
    // been generated. The used metadata is saved a separate file for
    // compilation.
    std::string consts = "";
    for (auto& it : this->imports()) {
      if (!std::strcmp(it->type_key(), "c")) {
        // TODO(zhiics) Maybe we need to store the list of required
        // variales in the CSourceModule so that we can validate the
        // existence of the variable and feed it only with the required
        // ones.
        CSourceMetadataInitializer c_init(metadata_);
        consts += c_init.Init();
        consts += "\n";
      }
    }
    if (consts != "") {
      std::string fmt = GetFileFormat(file_name, format);
      CHECK_EQ(fmt, "h") << "Can only save to .h file";
      SaveBinaryToFile(file_name, consts);
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    std::vector<std::string> variables;
    std::vector<runtime::NDArray> metadata;
    for (const auto& it : metadata_) {
      String var_name = it.first;
      variables.push_back(var_name.operator std::string());
      metadata.push_back(it.second);
    }

    // Save all variables in the function.
    stream->Write(variables);
    // Save all constant data.
    uint64_t sz = static_cast<uint64_t>(metadata.size());
    stream->Write(sz);
    for (uint64_t i = 0; i < sz; i++) {
      metadata[i].Save(stream);
    }
  }

  static runtime::Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);

    // Load the variables.
    std::vector<std::string> variables;
    CHECK(stream->Read(&variables)) << "Loading variables failed";
    uint64_t sz;
    CHECK(stream->Read(&sz, sizeof(sz))) << "Loading medata size failed";
    CHECK_EQ(static_cast<size_t>(sz), variables.size())
        << "The number of variables and ndarray counts must match";
    // Load the list of ndarray.
    std::vector<runtime::NDArray> metadata;
    for (uint64_t i = 0; i < sz; i++) {
      tvm::runtime::NDArray temp;
      temp.Load(stream);
      metadata.push_back(temp);
    }

    Array<String> vars;
    Array<runtime::NDArray> consts;
    for (size_t i = 0; i < variables.size(); i++) {
      vars.push_back(variables[i]);
      consts.push_back(metadata[i]);
    }
    auto n = runtime::make_object<ModuleInitWrapper>(vars, consts);
    return runtime::Module(n);
  }

 private:
  /*!
   * \brief Record if a module is initialized. It is needed by imported
   * modules using execution engine.
   */
  std::unordered_map<std::string, bool> initialized_;
  /*! \brief Variable name to NDArray mapping. */
  StringNDArrayMap metadata_;
};

runtime::Module ModuleInitWrapperCreate(Array<String> variables, Array<runtime::NDArray> metadata) {
  auto n = make_object<ModuleInitWrapper>(variables, metadata);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.ModuleInitWrapper").set_body_typed(ModuleInitWrapperCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_module_init")
    .set_body_typed(ModuleInitWrapper::LoadFromBinary);
}  // namespace runtime
}  // namespace tvm

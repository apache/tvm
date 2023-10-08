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
 * \file runtime/static_library.cc
 * \brief Represents a generic '.o' static library which can be linked into the final output
 * dynamic library by export_library.
 */
#include "./static_library.h"

#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <iostream>

#include "file_utils.h"

namespace tvm {
namespace runtime {

namespace {

/*!
 * \brief A '.o' library which can be linked into the final output library by export_library.
 * Can be used by external codegen tools which can produce a ready-to-link artifact.
 */
class StaticLibraryNode final : public runtime::ModuleNode {
 public:
  ~StaticLibraryNode() override = default;

  const char* type_key() const final { return "static_library"; }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_func_names") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = func_names_; });
    } else {
      return {};
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(data_);
    std::vector<std::string> func_names;
    for (const auto func_name : func_names_) func_names.push_back(func_name);
    stream->Write(func_names);
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    auto n = make_object<StaticLibraryNode>();
    // load data
    std::string data;
    ICHECK(stream->Read(&data)) << "Loading data failed";
    n->data_ = std::move(data);

    // load func names
    std::vector<std::string> func_names;
    ICHECK(stream->Read(&func_names)) << "Loading func names failed";
    for (auto func_name : func_names) n->func_names_.push_back(String(func_name));

    return Module(n);
  }

  void SaveToFile(const String& file_name, const String& format) final {
    VLOG(0) << "Saving static library of " << data_.size() << " bytes implementing " << FuncNames()
            << " to '" << file_name << "'";
    SaveBinaryToFile(file_name, data_);
  }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const override {
    return runtime::ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kDSOExportable;
  }

  bool ImplementsFunction(const String& name, bool query_imports) final {
    return std::find(func_names_.begin(), func_names_.end(), name) != func_names_.end();
  }

  std::string FuncNames() {
    std::ostringstream os;
    os << "[";
    bool first = true;
    for (const auto& func_name : func_names_) {
      if (first) {
        first = false;
      } else {
        os << ", ";
      }
      os << "'" << func_name << "'";
    }
    os << "]";
    return os.str();
  }

  /*! \brief Contents of the object file. */
  std::string data_;
  /*! \brief Function names exported by the above. */
  Array<String> func_names_;
};

}  // namespace

Module LoadStaticLibrary(const std::string& filename, Array<String> func_names) {
  auto node = make_object<StaticLibraryNode>();
  LoadBinaryFromFile(filename, &node->data_);
  node->func_names_ = std::move(func_names);
  VLOG(0) << "Loaded static library from '" << filename << "' implementing " << node->FuncNames();
  return Module(node);
}

TVM_REGISTER_GLOBAL("runtime.ModuleLoadStaticLibrary").set_body_typed(LoadStaticLibrary);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_static_library")
    .set_body_typed(StaticLibraryNode::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm

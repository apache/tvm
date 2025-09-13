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

#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/module.h>

#include <iostream>

#include "file_utils.h"

namespace tvm {
namespace runtime {

namespace {

/*!
 * \brief A '.o' library which can be linked into the final output library by export_library.
 * Can be used by external codegen tools which can produce a ready-to-link artifact.
 */
class StaticLibraryNode final : public ffi::ModuleObj {
 public:
  const char* kind() const final { return "static_library"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    const ObjectPtr<Object>& sptr_to_self = ffi::GetObjectPtr<Object>(this);
    if (name == "get_func_names") {
      return ffi::Function(
          [sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) { *rv = func_names_; });
    } else {
      return std::nullopt;
    }
  }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    dmlc::MemoryStringStream ms(&buffer);
    dmlc::Stream* stream = &ms;
    stream->Write(data_);
    std::vector<std::string> func_names;
    for (const auto func_name : func_names_) func_names.push_back(func_name);
    stream->Write(func_names);
    return ffi::Bytes(buffer);
  }

  static ffi::Module LoadFromBytes(ffi::Bytes bytes) {
    dmlc::MemoryFixedSizeStream ms(const_cast<char*>(bytes.data()), bytes.size());
    dmlc::Stream* stream = &ms;
    auto n = ffi::make_object<StaticLibraryNode>();
    // load data
    std::string data;
    ICHECK(stream->Read(&data)) << "Loading data failed";
    n->data_ = std::move(data);

    // load func names
    std::vector<std::string> func_names;
    ICHECK(stream->Read(&func_names)) << "Loading func names failed";
    for (auto func_name : func_names) n->func_names_.push_back(ffi::String(func_name));

    return ffi::Module(n);
  }

  void WriteToFile(const ffi::String& file_name, const ffi::String& format) const final {
    VLOG(0) << "Saving static library of " << data_.size() << " bytes implementing " << FuncNames()
            << " to '" << file_name << "'";
    SaveBinaryToFile(file_name, data_);
  }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const override {
    return ffi::Module::kBinarySerializable | ffi::Module::kCompilationExportable;
  }

  bool ImplementsFunction(const ffi::String& name) final {
    return std::find(func_names_.begin(), func_names_.end(), name) != func_names_.end();
  }

  std::string FuncNames() const {
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
  ffi::Array<ffi::String> func_names_;
};

}  // namespace

ffi::Module LoadStaticLibrary(const std::string& filename, ffi::Array<ffi::String> func_names) {
  auto node = ffi::make_object<StaticLibraryNode>();
  LoadBinaryFromFile(filename, &node->data_);
  node->func_names_ = std::move(func_names);
  VLOG(0) << "Loaded static library from '" << filename << "' implementing " << node->FuncNames();
  return ffi::Module(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.ModuleLoadStaticLibrary", LoadStaticLibrary)
      .def("ffi.Module.load_from_bytes.static_library", StaticLibraryNode::LoadFromBytes);
}

}  // namespace runtime
}  // namespace tvm

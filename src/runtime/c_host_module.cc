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
 * \file c_module.cc
 * \brief C module. Note this module is used for simple tasks on host machine
 *        (for example, launch GPU kernels). Performance is not the priority,
 *        but to allow users to use TVM without llvm dependency.
 * \note This module is different from source_module and is NOT designed for
 *       AOT or legacy micro usecases.
 */

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "file_utils.h"
#include "library_module.h"
#include "tvm/runtime/container/optional.h"

namespace tvm {
namespace runtime {

class CModuleNode : public runtime::ModuleNode {
 public:
  explicit CModuleNode(std::string c_source) : c_source_(c_source) {}

  ~CModuleNode() = default;

  const char* type_key() const final { return "c"; }

  String GetSource(const String& format) final { return c_source_; };

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;

  String GetFormat() final { return "c"; }

  void SaveToFile(const String& file_name, const String& format) final;

  int GetPropertyMask() const override {
    return runtime::ModulePropertyMask::kRunnable | runtime::ModulePropertyMask::kDSOExportable;
  }

 private:
  void Compile();
  /*! \brief The C source code */
  std::string c_source_;
  /*! \brief The library module */
  Optional<Module> compiled_lib_ = NullOpt;
};

void CModuleNode::SaveToFile(const String& file_name, const String& format) {
  std::string fmt = runtime::GetFileFormat(file_name, format);
  std::string meta_file = runtime::GetMetaFilePath(file_name);
  if (fmt == "c") {
    SaveBinaryToFile(file_name, c_source_);
  } else {
    LOG(FATAL) << "Unsupported format: " << fmt;
  }
}

PackedFunc CModuleNode::GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
  if (!compiled_lib_.defined()) {
    Compile();
  }
  return compiled_lib_.value()->GetFunction(name, sptr_to_self);
}

void CModuleNode::Compile() {
  ICHECK(!compiled_lib_.defined()) << "CModuleNode::Compile() should be called only once";
  if (const auto* f = Registry::Get("tvm_callback_c_compile")) {
    std::string binary = (*f)(c_source_, /*target=*/"c").operator std::string();
    std::string tmp_file_name = std::string(std::tmpnam(nullptr)) + ".so";
    std::ofstream tmp_file(tmp_file_name, std::ios::binary);
    tmp_file.write(binary.c_str(), binary.size());
    tmp_file.close();
    ObjectPtr<Library> lib = CreateDSOLibraryObject(tmp_file_name);
    compiled_lib_ = CreateModuleFromLibrary(lib);
    std::remove(tmp_file_name.c_str());
  } else {
    LOG(FATAL) << "tvm_callback_c_compile not found";
  }
}

Module CModuleCreate(std::string c_source) {
  auto n = make_object<CModuleNode>(c_source);
  return runtime::Module(n);
}

}  // namespace runtime
}  // namespace tvm

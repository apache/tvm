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
 * \file dso_module.cc
 * \brief Module to load from dynamic shared library.
 */
#include "dso_module.h"

#include <tvm/runtime/module.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <string>
#include "module_util.h"

namespace tvm {
namespace runtime {

PackedFunc DSOModuleNode::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  BackendPackedCFunc faddr;
  if (name == runtime::symbol::tvm_module_main) {
    const char* entry_name = reinterpret_cast<const char*>(
        GetSymbol(runtime::symbol::tvm_module_main));
    CHECK(entry_name!= nullptr)
        << "Symbol " << runtime::symbol::tvm_module_main << " is not presented";
    faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(entry_name));
  } else {
    faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(name.c_str()));
  }
  if (faddr == nullptr) return PackedFunc();
  return WrapPackedFunc(faddr, sptr_to_self);
}

void DSOModuleNode::Init(const std::string& name) {
  Load(name);
  if (auto *ctx_addr =
      reinterpret_cast<void**>(GetSymbol(runtime::symbol::tvm_module_ctx))) {
    *ctx_addr = this;
  }

  PackedFunc GetFunction(
      const std::string& name,
      const ObjectPtr<Object>& sptr_to_self) final {
    BackendPackedCFunc faddr;
    if (name == runtime::symbol::tvm_module_main) {
      const char* entry_name = reinterpret_cast<const char*>(
          GetSymbol(runtime::symbol::tvm_module_main));
      CHECK(entry_name!= nullptr)
          << "Symbol " << runtime::symbol::tvm_module_main << " is not presented";
      faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(entry_name));
    } else {
      faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(name.c_str()));
    }
    if (faddr == nullptr) return PackedFunc();
    return WrapPackedFunc(faddr, sptr_to_self);
  }
}

TVM_REGISTER_GLOBAL("module.loadfile_so")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    auto n = make_object<DSOModuleNode>();
    n->Init(args[0]);
    *rv = runtime::Module(n);
  });
}  // namespace runtime
}  // namespace tvm

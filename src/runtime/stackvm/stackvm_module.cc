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
 *  Copyright (c) 2017 by Contributors
 * \file stackvm_module.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>
#include <dmlc/memory_io.h>
#include <memory>
#include <utility>
#include <unordered_map>
#include "stackvm_module.h"
#include "../file_util.h"
#include "../module_util.h"

namespace tvm {
namespace runtime {

class StackVMModuleNode : public runtime::ModuleNode {
 public:
  const char* type_key() const {
    return "stackvm";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    if (name == runtime::symbol::tvm_module_main) {
      return GetFunction(entry_func_, sptr_to_self);
    }
    auto it = fmap_.find(name);
    if (it == fmap_.end()) return PackedFunc();
    const StackVM& vm = it->second;
    // capture sptr_to_self to keep module node alive.
    return PackedFunc([vm, sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        vm.Run(args, this);
      });
  }

  std::string GetSource(const std::string& format) final {
    std::ostringstream os;
    for (const auto& kv : fmap_) {
      os << "Function: " << kv.first << '\n';
      os << kv.second;
    }
    return os.str();
  }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string data, mblob;
    dmlc::MemoryStringStream writer(&data);
    dmlc::Stream* strm = &writer;
    strm->Write(fmap_);
    strm->Write(entry_func_);
    // also save imports
    uint64_t num_imports = static_cast<uint64_t>(imports_.size());
    strm->Write(num_imports);

    for (runtime::Module im : imports_) {
      CHECK_EQ(im->imports().size(), 0U)
          << "Only support simply one-level hierarchy";
      std::string tkey = im->type_key();
      strm->Write(tkey);
      LOG(INFO) << "save " << tkey;
      im->SaveToBinary(strm);
      LOG(INFO) << "FInish save " << tkey;
    }
    SaveBinaryToFile(file_name, data);
  }

  static Module Create(std::unordered_map<std::string, StackVM> fmap,
                       std::string entry_func) {
    std::shared_ptr<StackVMModuleNode> n =
        std::make_shared<StackVMModuleNode>();
    n->fmap_ = std::move(fmap);
    n->entry_func_ = std::move(entry_func);
    return Module(n);
  }

  static Module Load(dmlc::Stream* strm) {
    std::unordered_map<std::string, StackVM> fmap;
    std::string entry_func, data;
    strm->Read(&fmap);
    strm->Read(&entry_func);
    std::shared_ptr<StackVMModuleNode> n =
        std::make_shared<StackVMModuleNode>();
    n->fmap_ = std::move(fmap);
    n->entry_func_ = std::move(entry_func);
    uint64_t num_imports;
    strm->Read(&num_imports);
    for (uint64_t i = 0; i < num_imports; ++i) {
      std::string tkey;
      CHECK(strm->Read(&tkey));
      std::string fkey = "module.loadbinary_" + tkey;
      const PackedFunc* f = Registry::Get(fkey);
      CHECK(f != nullptr)
          << "Loader of " << tkey << "("
          << fkey << ") is not presented.";
      Module m = (*f)(static_cast<void*>(strm));
      n->imports_.emplace_back(std::move(m));
    }
    return Module(n);
  }

  static Module LoadFromFile(std::string file_name,
                             std::string format) {
    std::string data;
    LoadBinaryFromFile(file_name, &data);
    dmlc::MemoryStringStream reader(&data);
    return Load(&reader);
  }

 private:
  // internal function map
  std::unordered_map<std::string, StackVM> fmap_;
  // entry function.
  std::string entry_func_;
};

Module StackVMModuleCreate(std::unordered_map<std::string, StackVM> fmap,
                           std::string entry_func) {
  return StackVMModuleNode::Create(fmap, entry_func);
}

TVM_REGISTER_GLOBAL("module.loadfile_stackvm")
.set_body_typed(StackVMModuleNode::LoadFromFile);

}  // namespace runtime
}  // namespace tvm
